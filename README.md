# HackRx Bill Extraction API (FastAPI + Gemini)

This project implements the **HackRx Bill Extraction API** using **FastAPI**, **Tesseract OCR**, and **Google Gemini (Generative AI)**. Given a publicly accessible **image or PDF URL** of a bill/invoice, the API downloads the document, converts each page to an image (for PDFs), runs OCR to extract raw text, and then uses Gemini in JSON mode to extract **page-wise line items**. For each page it classifies the `page_type` (`Bill Detail`, `Final Bill`, or `Pharmacy`) and returns structured `bill_items` containing `item_name`, `item_amount`, `item_rate`, and `item_quantity`. The service also deduplicates repeated entries, filters out totals/subtotals, and enforces basic consistency between `rate × quantity` and `amount` to avoid double counting and gross OCR/LLM mistakes. Token usage from Gemini is tracked and returned so that you can monitor LLM consumption per request.

---

## Features

This API automatically downloads bill documents from a URL, handles both single-page images and multi-page PDFs, converts pages to images, runs Tesseract OCR to extract text, and uses Google Gemini in JSON mode to generate clean, structured line items per page; it classifies each page type, filters out totals and taxes to avoid double counting, lightly cleans item names, enforces basic arithmetic consistency between item rate, quantity, and amount, aggregates all items across pages with deduplication, and finally returns a JSON response that matches the HackRx submission schema while also reporting cumulative LLM token usage for observability.

---

## Tech Stack

- **Backend**: FastAPI (Python)
- **OCR**: Tesseract (`pytesseract`)
- **PDF to Image**: `pdf2image`
- **LLM**: Google Gemini via `google-genai`
- **Environment Management**: `python-dotenv` and virtual environment (`venv`)

---

## Project Structure (key files)

- `main.py` – FastAPI app implementing `/extract-bill-data`
- `requirements.txt` – Python dependencies
- `.env` – Environment variables (not committed; you create this)
- (Optional) `README.md` – This file

---

## API Specification

### Endpoint

`POST /extract-bill-data`  
Content-Type: `application/json`

### Request Body

```json
{
  "document": "https://example.com/path/to/bill_or_invoice.pdf"
}
```

### Successful Response (200 OK)
```json
{
  "is_success": true,
  "token_usage": {
    "total_tokens": 6224,
    "input_tokens": 1365,
    "output_tokens": 1930
  },
  "data": {
    "pagewise_line_items": [
      {
        "page_no": "1",
        "page_type": "Bill Detail",
        "bill_items": [
          {
            "item_name": "Consultation Charge | DR PREETHI",
            "item_amount": 300.0,
            "item_rate": 300.0,
            "item_quantity": 1.0
          }
          // ...
        ]
      }
      // ...
    ],
    "total_item_count": 30
  }
}
```
### Failure Response (200 OK, is_success=false)
```json
{
  "is_success": false,
  "token_usage": {
    "total_tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0
  },
  "data": null
}
```
## Setup Instructions
### 1. Clone the repository

```bash 
git clone https://github.com/mayukhabhigyandas/Bajaj-Health-Datathon.git

cd Bajaj-Health-Datathon
```

### 2. Create & activate a Virtual Environment

```bash
# Create venv
python -m venv venv (Windows)
python3 -m venv venv (macOS/Linux)

# Activate venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```
### 3. Install Dependencies
```bash 
pip install -r requirements.txt

sudo apt install -y tesseract-ocr poppler-utils(For Windows download the .exe and add to local path)
```

### 4. Run the Application
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Configure .env (do not commit the API KEY)
Create a .env file in the same folder as main.py:
```bash 
GEMINI_API_KEY=your_gemini_api_key_here
```

## Testing the API
### Using curl

```bash 
curl -X POST "http://localhost:8000/extract-bill-data" \
  -H "Content-Type: application/json" \
  -d '{
    "document": "https://hackrx.blob.core.windows.net/assets/datathon-IIT/sample_2.png?..."
  }'
```
