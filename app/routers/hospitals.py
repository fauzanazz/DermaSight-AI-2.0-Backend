from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from ..db import get_db
from ..schemas import HospitalOut
from ..services.hospitals import find_hospitals

router = APIRouter(prefix="/api/v1", tags=["hospitals"])

@router.get("/hospitals", response_model=list[HospitalOut])
async def hospitals(
    lat: float = Query(..., description="Latitude coordinate"),
    lng: float = Query(..., description="Longitude coordinate"), 
    condition: str | None = Query(None, description="ICD-10 condition code filter"),
    radius_km: float = Query(20.0, ge=1.0, le=100.0, description="Search radius in kilometers"),
    db: AsyncSession = Depends(get_db)
):
    if not -90 <= lat <= 90:
        raise HTTPException(status_code=400, detail="Invalid latitude")
    if not -180 <= lng <= 180:
        raise HTTPException(status_code=400, detail="Invalid longitude")
    
    return await find_hospitals(db, lat, lng, condition, radius_km)