from pydantic import BaseModel, Field
from typing import List, Optional, Literal

Severity = Literal['mild', 'moderate', 'severe']

class ConditionProb(BaseModel):
    label: str
    icd10: Optional[str] = None
    confidence: float = Field(ge=0, le=1)

class DiagnosisResponse(BaseModel):
    topCondition: ConditionProb
    conditions: List[ConditionProb]
    severity: Severity
    preMedication: List[str]
    advice: Optional[str] = None
    _meta: dict | None = None

class HospitalOut(BaseModel):
    id: str
    name: str
    lat: float
    lng: float
    distanceKm: float | None = None
    equipments: List[str] = []
    specialties: List[str] = []
    qualifiedFor: List[str] = []
    phone: str | None = None
    address: str | None = None

class BookingCreate(BaseModel):
    hospitalId: str
    conditionCode: Optional[str] = None
    patientName: str
    phone: str
    notes: Optional[str] = None
    preferredAt: Optional[str] = None

class BookingOut(BaseModel):
    bookingId: str
    status: Literal['pending','confirmed','rejected']
    estimatedWaitMin: Optional[int] = None