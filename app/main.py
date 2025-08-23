from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .routers import diagnose, hospitals, bookings

app = FastAPI(
    title="DermaSight AI API", 
    version="1.0.0",
    description="Backend API for DermaSight AI skin condition analysis and hospital booking system"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(',')],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(diagnose.router)
app.include_router(hospitals.router)
app.include_router(bookings.router)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "dermasight-api"}

@app.get("/")
async def root():
    return {
        "message": "DermaSight AI Backend API",
        "version": "1.0.0",
        "docs": "/docs"
    }