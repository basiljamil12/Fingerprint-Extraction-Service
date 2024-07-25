from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.controllers.fingerprint_controller import router as fingerprint_router

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(fingerprint_router, prefix="/api/v1/fingerprints")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)
