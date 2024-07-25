from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.services.fingerprint_extract_service import process_fingerprint
import httpx
from io import BytesIO
import asyncio

router = APIRouter()

class FingerprintRequest(BaseModel):
    userId: str
    finger: str  # URL of the image

# Create a semaphore to limit concurrency
semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent tasks

@router.post("/extract")
async def extract_fingerprints(request: FingerprintRequest):
    user_id = request.userId
    image_url = request.finger
    
    if not user_id:
        raise HTTPException(status_code=400, detail="No userId provided in JSON data")
    
    if not image_url:
        raise HTTPException(status_code=400, detail="No finger URL provided in JSON data")
    
    async with semaphore:  # Acquire semaphore
        try:
            # Asynchronously download the image from the provided URL
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url)
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail="Failed to retrieve image from URL")
            
            # Process the image
            image_data = BytesIO(response.content)  # Use BytesIO to handle the image in-memory
            data = process_fingerprint(image_data, user_id)
            
            return JSONResponse(content={'success': True, 'data': data, 'message': 'Fingerprint processing completed'}, status_code=200)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
