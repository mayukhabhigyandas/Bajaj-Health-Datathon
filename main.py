import io
import os
import traceback
from typing import List, Optional, Tuple
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
load_dotenv()  

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from PIL import Image

# OCR
import pytesseract

# PDF → images
from pdf2image import convert_from_bytes

from google import genai
from google.genai import types as genai_types


# Pydantic models (request/response schemas)

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float


class PageItems(BaseModel):
    page_no: str
    page_type: str  # "Bill Detail" | "Final Bill" | "Pharmacy"
    bill_items: List[BillItem]


class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int


class DataPayload(BaseModel):
    pagewise_line_items: List[PageItems]
    total_item_count: int


class ExtractResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: Optional[DataPayload]


class ExtractRequest(BaseModel):
    document: HttpUrl


# Internal schema for LLM structured output (for one page)
# This is used as Gemini response_schema (JSON mode).

class LLMPageResult(BaseModel):
    page_no: str
    page_type: str
    bill_items: List[BillItem]


# FastAPI app & global Gemini client

app = FastAPI(title="HackRx Bill Extraction API (Gemini)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Read API key from environment and create global client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    # Fail early with a clear error instead of the cryptic ValueError
    raise RuntimeError(
        "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY "
        "in your .env or environment before running the app."
    )

GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)



# Helpers – Document download & page splitting

async def download_document(url: str) -> bytes:
    """Download document from URL and return raw bytes."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


def is_pdf(file_bytes: bytes) -> bool:
    """Simple check: does this look like a PDF file?"""
    return file_bytes[:4] == b"%PDF"


def bytes_to_images(file_bytes: bytes) -> List[Image.Image]:
    """
    Convert input bytes to list of PIL.Image pages.
    - If PDF → use pdf2image.convert_from_bytes.
    - Else → treat as a single image.
    """
    if is_pdf(file_bytes):
        # PDF with potentially multiple pages
        images = convert_from_bytes(file_bytes, dpi=300)
        return images
    else:
        # Single image
        img = Image.open(io.BytesIO(file_bytes))
        # Ensure it's in RGB for OCR
        if img.mode != "RGB":
            img = img.convert("RGB")
        return [img]


# Helpers – OCR

def run_ocr(page_image: Image.Image) -> str:
    """
    Run OCR on a page image and return extracted text.
    You can add preprocessing here if needed.
    """
    custom_config = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(page_image, config=custom_config)
    return text


# Helpers – Gemini prompts

def build_llm_prompts(page_no: int, ocr_text: str) -> Tuple[str, str]:
    """
    Returns (system_instruction, user_prompt) for the Gemini call.
    We use system_instruction + user content as per google-genai docs.
    """
    system_instruction = """
You are an information extraction engine for hospital and pharmacy bills.

You will be given OCR text from a single page of a bill. Your job is to:

1. Determine the type of the page:
   - "Pharmacy" if the page contains medicines, drug codes, or pharmacy items.
   - "Final Bill" if it is a summary page showing total or net payable amounts.
   - "Bill Detail" for all other detailed pages that contain line items.

2. Extract only the BILL LINE ITEMS that represent products or services.
   - DO NOT include rows for Subtotal, Total, Grand Total, Net Payable,
     Taxes (CGST, SGST, IGST, VAT), Discounts, Rounding, or similar summary rows.
   - DO NOT double count items.
   - If a page only contains totals and no actual line items, bill_items must be an empty list.

3. For each valid line item, extract:
   - item_name: string, EXACTLY as it appears in the bill (do not normalize).
   - item_rate: numeric rate as shown in the bill.
   - item_quantity: numeric quantity.
   - item_amount: net amount for that line item AFTER discounts, as shown in the bill.

You must respond in JSON that matches the given response schema exactly.
""".strip()

    user_prompt = f"""
Page number: {page_no}

OCR text:
\"\"\"text
{ocr_text}
\"\"\"
""".strip()

    return system_instruction, user_prompt


def extract_page_with_gemini(
    page_no: int,
    ocr_text: str,
    model_name: str = "gemini-2.5-flash",
) -> Tuple[PageItems, TokenUsage]:
    """
    Call Gemini to convert OCR text into structured PageItems.
    Uses JSON mode with response_schema=LLMPageResult and returns Pydantic objects.
    """
    client = GEMINI_CLIENT  # reuse global client
    system_instruction, user_prompt = build_llm_prompts(page_no, ocr_text)

    # JSON mode: response_mime_type="application/json"
    # response_schema = LLMPageResult ensures parsed structured output.
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=LLMPageResult,
            temperature=0.0,
        ),
    )

    # response.parsed is already an instance of LLMPageResult
    llm_result: LLMPageResult = response.parsed  # type: ignore

    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        # Usage fields from google-genai
        total_tokens = usage.total_token_count or 0
        prompt_tokens = usage.prompt_token_count or 0
        completion_tokens = usage.candidates_token_count or 0
    else:
        total_tokens = prompt_tokens = completion_tokens = 0

    page_items = PageItems(
        page_no=llm_result.page_no,
        page_type=llm_result.page_type,
        bill_items=llm_result.bill_items,
    )

    token_usage = TokenUsage(
        total_tokens=total_tokens,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
    )

    return page_items, token_usage


# Helpers – Post-processing & de-duplication

def deduplicate_page_items(pages: List[PageItems]) -> List[PageItems]:
    """
    Simple de-duplication across pages using a set of signatures.
    Avoids double counting identical items that might appear on multiple pages.
    Also filters obvious total/summary rows by name as extra safety.
    """
    seen = set()
    new_pages: List[PageItems] = []

    for page in pages:
        filtered_items: List[BillItem] = []
        for it in page.bill_items:
            upper_name = it.item_name.upper()

            # Skip obvious total/summary rows by name
            if any(
                kw in upper_name
                for kw in [
                    "TOTAL",
                    "SUBTOTAL",
                    "NET PAYABLE",
                    "ROUND OFF",
                    "BALANCE",
                    "DISCOUNT",
                    "CGST",
                    "SGST",
                    "IGST",
                    "VAT",
                    "TAX",
                ]
            ):
                continue

            sig = (
                page.page_no,
                upper_name.strip(),
                float(it.item_rate),
                float(it.item_quantity),
                float(it.item_amount),
            )

            if sig in seen:
                continue

            seen.add(sig)
            filtered_items.append(it)

        new_pages.append(
            PageItems(
                page_no=page.page_no,
                page_type=page.page_type,
                bill_items=filtered_items,
            )
        )

    return new_pages


# Main endpoint

@app.post("/extract-bill-data", response_model=ExtractResponse)
async def extract_bill_data(req: ExtractRequest) -> ExtractResponse:
    """
    Main API endpoint required by the problem statement.
    Always returns HTTP 200 with is_success true/false as per spec.
    """
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    try:
        # 1) Download document
        file_bytes = await download_document(str(req.document))

        # 2) Convert to page images
        page_images = bytes_to_images(file_bytes)

        all_page_items: List[PageItems] = []

        # 3) Per-page OCR + Gemini extraction
        for idx, img in enumerate(page_images, start=1):
            ocr_text = run_ocr(img)

            page_items, usage = extract_page_with_gemini(idx, ocr_text)

            total_tokens += usage.total_tokens
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens

            all_page_items.append(page_items)

        # 4) De-duplicate items / clean
        all_page_items = deduplicate_page_items(all_page_items)

        # 5) Total item count
        total_item_count = sum(len(p.bill_items) for p in all_page_items)

        # 6) Build success response
        return ExtractResponse(
            is_success=True,
            token_usage=TokenUsage(
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            data=DataPayload(
                pagewise_line_items=all_page_items,
                total_item_count=total_item_count,
            ),
        )

    except Exception:
        # Log server-side
        traceback.print_exc()

        # On failure, keep schema but mark is_success = False
        return ExtractResponse(
            is_success=False,
            token_usage=TokenUsage(
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            data=None,
        )

@app.on_event("shutdown")
def shutdown_event():
    try:
        GEMINI_CLIENT.close()
    except Exception:
        pass


@app.get("/")
async def root():
    return {"status": "ok", "message": "HackRx Bill Extraction API (Gemini) running"}
