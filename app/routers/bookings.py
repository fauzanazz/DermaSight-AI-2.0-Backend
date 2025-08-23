from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..db import get_db
from ..models import Booking, BookingStatus, Hospital
from ..schemas import BookingCreate, BookingOut
import re

router = APIRouter(prefix="/api/v1", tags=["bookings"])

def validate_phone(phone: str) -> bool:
    phone_pattern = r'^[\+]?[1-9][\d]{0,15}$'
    return bool(re.match(phone_pattern, phone.replace('-', '').replace(' ', '')))

@router.post("/bookings", response_model=BookingOut, status_code=201)
async def create_booking(body: BookingCreate, db: AsyncSession = Depends(get_db)):
    if not body.patientName or len(body.patientName.strip()) < 2:
        raise HTTPException(status_code=422, detail="Patient name must be at least 2 characters")
    
    if not body.phone or not validate_phone(body.phone):
        raise HTTPException(status_code=422, detail="Invalid phone number format")
    
    hosp = await db.scalar(select(Hospital).where(Hospital.id == body.hospitalId))
    if not hosp:
        raise HTTPException(status_code=404, detail="Hospital not found")

    b = Booking(
        hospital_id=body.hospitalId,
        condition_code=body.conditionCode,
        patient_name=body.patientName.strip(),
        phone=body.phone.strip(),
        notes=body.notes.strip() if body.notes else None,
        preferred_at=body.preferredAt,
        status=BookingStatus.pending
    )
    db.add(b)
    await db.commit()
    await db.refresh(b)
    
    return BookingOut(
        bookingId=str(b.id), 
        status=b.status.value,
        estimatedWaitMin=30
    )