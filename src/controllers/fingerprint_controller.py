from fastapi import APIRouter, HTTPException, UploadFile, File,Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.services.fingerprint_extract_service import process_fingerprint
import httpx
from io import BytesIO
import asyncio

router = APIRouter()

# Create a semaphore to limit concurrency
semaphore = asyncio.Semaphore(10)

@router.post("/extract")
async def extract_fingerprints( file: UploadFile = File(...)):

    if not file:
        raise HTTPException(status_code=400, detail="No file provided in form data")

    async with semaphore:  # Acquire semaphore
        try:
            # Read the file contents
            image_data = await file.read()
            image_stream = BytesIO(image_data)

            # Process the image
            data = process_fingerprint(image_stream)

            return JSONResponse(content={'success': True, 'data': data, 'message': 'Fingerprint processing completed'}, status_code=200)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
