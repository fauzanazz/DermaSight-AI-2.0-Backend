from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from sqlalchemy.orm import selectinload
from geoalchemy2.shape import to_shape
from ..models import Hospital, HospitalCondition
from ..schemas import HospitalOut
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

async def find_hospitals(
    db: AsyncSession, 
    lat: float, 
    lng: float, 
    icd10: str | None, 
    radius_km: float = 20.0
) -> list[HospitalOut]:
    
    stmt = (
        select(Hospital)
        .options(
            selectinload(Hospital.equipments),
            selectinload(Hospital.specialties), 
            selectinload(Hospital.conditions)
        )
    )
    
    if icd10:
        stmt = stmt.join(HospitalCondition).where(HospitalCondition.icd10 == icd10)

    result = await db.execute(stmt)
    hospitals = result.scalars().all()

    out: list[HospitalOut] = []
    for h in hospitals:
        shape = to_shape(h.location)
        h_lat, h_lng = shape.y, shape.x
        
        distance_km = haversine_distance(lat, lng, h_lat, h_lng)
        
        if distance_km <= radius_km:
            out.append(HospitalOut(
                id=str(h.id),
                name=h.name,
                lat=h_lat,
                lng=h_lng,
                distanceKm=round(distance_km, 2),
                equipments=[e.name for e in h.equipments],
                specialties=[s.name for s in h.specialties],
                qualifiedFor=[c.icd10 for c in h.conditions],
                phone=h.phone,
                address=h.address
            ))
    
    out.sort(key=lambda x: x.distanceKm or float('inf'))
    return out