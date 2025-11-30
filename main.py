import io
import os
import asyncio
import traceback
from typing import List, Optional, Tuple

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, HTTPException
from fastapi.concurrency import run_in_threadpool

from dotenv import load_dotenv
load_dotenv()

import httpx
from pydantic import BaseModel
from PIL import Image

# OCR
import pytesseract

from pdf2image import convert_from_bytes

from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors


POPPLER_PATH = os.getenv("POPPLER_PATH")
TESSERACT_CMD = os.getenv("TESSERACT_CMD")

if not POPPLER_PATH:
    raise RuntimeError(
        "POPPLER_PATH not set. Add POPPLER_PATH to your .env "
        "pointing to the Poppler 'bin' directory."
    )

if not TESSERACT_CMD:
    raise RuntimeError(
        "TESSERACT_CMD not set. Add TESSERACT_CMD to your .env "
        "pointing to tesseract.exe (e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe)."
    )

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float = 0.0
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


# Internal schema for LLM structured output
class LLMPageResult(BaseModel):
    page_no: str
    page_type: str
    bill_items: List[BillItem]


app = FastAPI(title="HackRx Bill Extraction API (Gemini)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY "
        "in your .env or environment before running the app."
    )

GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)

# Reusable HTTP client for document download
HTTP_CLIENT = httpx.AsyncClient(timeout=60)


async def download_document(url: str) -> bytes:
    """Download document from URL and return raw bytes."""
    resp = await HTTP_CLIENT.get(url)
    resp.raise_for_status()
    return resp.content


def is_pdf(file_bytes: bytes) -> bool:
    """Simple check: does this look like a PDF file?"""
    return file_bytes[:4] == b"%PDF"


def bytes_to_images(file_bytes: bytes) -> List[Image.Image]:
    """
    Convert input bytes to list of PIL.Image pages.
    - If PDF → use pdf2image.convert_from_bytes (Poppler required).
    - Else → treat as a single image.
    """
    if is_pdf(file_bytes):
        images = convert_from_bytes(
            file_bytes,
            dpi=220,
            poppler_path=POPPLER_PATH,
        )
        return images
    else:
        img = Image.open(io.BytesIO(file_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return [img]


def run_ocr(page_image: Image.Image) -> str:
    """
    Run OCR on a page image and return extracted text.
    """
    # English only; adjust if you need more langs
    custom_config = r"--oem 3 --psm 6 -l eng"
    text = pytesseract.image_to_string(page_image, config=custom_config)
    return text


def has_rate_column(ocr_text: str) -> bool:
    """
    Detect if the page has a Rate/Price column at all.
    If this returns False, we will force all item_rate = 0 for that page.
    """
    text_upper = ocr_text.upper()

    # Common header keywords that indicate a rate/price column exists
    keywords = [
        " RATE ",       # surrounded by spaces
        "\nRATE ",      # start of line-like
        " RATE\n",
        "\nRATE\n",
        " RATE/UNIT",
        " UNIT RATE",
        " UNIT PRICE",
        " PRICE ",
        "\nPRICE ",
        " MRP ",
        "\nMRP ",
    ]

    return any(kw in text_upper for kw in keywords)


def build_llm_prompts(page_no: int, ocr_text: str) -> Tuple[str, str]:
    """
    Returns (system_instruction, user_prompt) for the Gemini call.
    Shorter prompt for speed & lower token cost.
    """
    system_instruction = """
You extract structured line items from hospital and pharmacy bills.

Given OCR text for ONE page:
1. Classify page_type:
   - "Pharmacy" → medicines/drugs/pharmacy items.
   - "Final Bill" → summary page with overall totals/net payable.
   - "Bill Detail" → other detailed pages with line items.
2. Extract ONLY real product/service line items:
   - EXCLUDE rows for Subtotal/Total/Grand Total/Net Payable,
     discounts, taxes (CGST, SGST, IGST, VAT), rounding, balance.
   - Do NOT double-count items.
   - If page has only totals and no items, bill_items = [].
3. For each line item return:
   - item_name: exactly as in bill.
   - item_rate: numeric. If the rate is NOT explicitly present in the OCR text for that line item, set item_rate = 0.
   - item_quantity: numeric.
   - item_amount: numeric, net amount after discounts.

Respond as JSON strictly matching the given schema.
""".strip()

    user_prompt = f"""
Page number: {page_no}

OCR text:
\"\"\"text
{ocr_text}
\"\"\""""
    return system_instruction, user_prompt.strip()


def extract_page_with_gemini(
    page_no: int,
    ocr_text: str,
    model_name: str = "gemini-2.5-flash",
) -> Tuple[PageItems, TokenUsage]:
    """
    Call Gemini to convert OCR text into structured PageItems.
    Uses JSON mode with response_schema=LLMPageResult and returns Pydantic objects.
    Includes graceful handling of quota/rate-limit errors.
    """
    client = GEMINI_CLIENT
    system_instruction, user_prompt = build_llm_prompts(page_no, ocr_text)

    try:
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
    except genai_errors.ClientError as e:
        # Quota/rate limit exceeded
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            raise HTTPException(
                status_code=429,
                detail="Gemini quota / rate limit exceeded. Please reduce request rate or upgrade quota.",
            )
        # Other Gemini client errors
        raise HTTPException(
            status_code=502,
            detail=f"Gemini ClientError: {e}",
        )
    except Exception as e:
        # Generic failure from SDK
        raise HTTPException(
            status_code=502,
            detail=f"Gemini call failed: {e}",
        )

    llm_result: LLMPageResult = response.parsed  # type: ignore

    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
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


async def process_page(idx: int, img: Image.Image) -> Tuple[PageItems, TokenUsage]:
    """
    Process a single page:
    - OCR in a threadpool
    - Gemini call in a threadpool
    - If the OCR text does NOT contain a Rate/Price column, force all item_rate = 0.
    """
    # OCR (CPU-bound) → offload to thread
    ocr_text = await run_in_threadpool(run_ocr, img)

    # Gemini call → offload to thread (blocking I/O / CPU on SDK)
    page_items, usage = await run_in_threadpool(extract_page_with_gemini, idx, ocr_text)

    # ---- NEW LOGIC: If the bill/page has NO rate column, set all item_rate = 0 ----
    if not has_rate_column(ocr_text):
        for it in page_items.bill_items:
            it.item_rate = 0.0

    return page_items, usage


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


@app.post("/extract-bill-data", response_model=ExtractResponse)
async def extract_bill_data(request: Request) -> ExtractResponse:
    """
    Single endpoint that accepts:

    1) application/json:
       {
         "document": "https://...."   # URL of PDF/image
       }

    2) multipart/form-data:
       - file: uploaded PDF/image
       - OR document: URL string
    """
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    document_url: Optional[str] = None
    file_bytes: Optional[bytes] = None

    try:
        content_type = request.headers.get("content-type", "")

        # A) JSON input (URL only)
        if "application/json" in content_type:
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="Invalid JSON body")
            document_url = body.get("document")

        # B) Multipart form-data (file or URL, or both)
        elif "multipart/form-data" in content_type:
            form = await request.form()
            if "document" in form:
                document_url = form.get("document")
            if "file" in form:
                uploaded_file = form.get("file")
                if uploaded_file is not None:
                    file_bytes = await uploaded_file.read()

        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported Content-Type. Use application/json or multipart/form-data.",
            )

        # Validate input presence
        if not document_url and not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'document' (URL) or 'file' (uploaded PDF/image).",
            )

        if document_url and file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Provide only ONE of 'document' (URL) or 'file', not both.",
            )

        # Download from URL if provided
        if document_url:
            if not (
                document_url.startswith("http://")
                or document_url.startswith("https://")
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid document URL. Must start with http:// or https://",
                )
            file_bytes = await download_document(document_url)

        # At this point, file_bytes is guaranteed not None
        assert file_bytes is not None, "file_bytes should not be None here"

        # 2) Convert to page images
        page_images = bytes_to_images(file_bytes)

        # Optional safety: limit pages to avoid huge bills
        # if len(page_images) > 20:
        #     page_images = page_images[:20]

        # 3) Per-page OCR + Gemini extraction SEQUENTIALLY
        all_page_items: List[PageItems] = []
        for idx, img in enumerate(page_images, start=1):
            page_items, usage = await process_page(idx, img)

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

    except HTTPException as e:
        traceback.print_exc()
        raise e
    except Exception:
        traceback.print_exc()
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
async def shutdown_event():
    try:
        if hasattr(GEMINI_CLIENT, "close"):
            GEMINI_CLIENT.close()
    except Exception:
        pass

    try:
        await HTTP_CLIENT.aclose()
    except Exception:
        pass


@app.get("/")
async def root():
    return {"status": "ok", "message": "HackRx Bill Extraction API (Gemini) running"}
