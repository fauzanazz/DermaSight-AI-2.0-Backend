import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from app.config import settings
from app.models import Hospital, HospitalSpecialty, HospitalEquipment, HospitalCondition
import uuid

async def seed_hospitals():
    """Seed the database with sample hospital data"""
    engine = create_async_engine(settings.database_url)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with AsyncSessionLocal() as session:
        hospitals_data = [
            {
                "id": str(uuid.uuid4()),
                "name": "Jakarta Dermatology Center",
                "address": "Jl. Sudirman No. 123, Jakarta Pusat",
                "phone": "+62-21-555-0001",
                "lat": -6.2088,
                "lng": 106.8456,
                "specialties": ["Dermatology", "Cosmetic Surgery"],
                "equipments": ["Digital Dermatoscope", "Photodynamic Therapy", "Laser Treatment"],
                "conditions": ["L20", "L40", "L21", "L57.0"]
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Skin Care Hospital Bandung",
                "address": "Jl. Asia Afrika No. 45, Bandung",
                "phone": "+62-22-555-0002", 
                "lat": -6.9175,
                "lng": 107.6191,
                "specialties": ["Dermatology", "Plastic Surgery"],
                "equipments": ["Cryotherapy", "Electrosurgery", "Biopsy Kit"],
                "conditions": ["L20", "L25", "L57.0"]
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Surabaya Medical Dermatology",
                "address": "Jl. Pemuda No. 78, Surabaya",
                "phone": "+62-31-555-0003",
                "lat": -7.2575,
                "lng": 112.7521,
                "specialties": ["Dermatology", "Allergy Treatment"],
                "equipments": ["Wood's Lamp", "Patch Testing Kit", "Digital Imaging"],
                "conditions": ["L20", "L40", "L25"]
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Bali Skin Clinic",
                "address": "Jl. Sunset Road No. 99, Denpasar",
                "phone": "+62-361-555-0004",
                "lat": -8.6705,
                "lng": 115.2126,
                "specialties": ["Dermatology", "Aesthetic Medicine"],
                "equipments": ["IPL Machine", "RF Treatment", "Chemical Peel"],
                "conditions": ["L21", "L57.0", "L25"]
            }
        ]
        
        for hosp_data in hospitals_data:
            hospital = Hospital(
                id=hosp_data["id"],
                name=hosp_data["name"],
                address=hosp_data["address"],
                phone=hosp_data["phone"],
                location=f"SRID=4326;POINT({hosp_data['lng']} {hosp_data['lat']})"
            )
            session.add(hospital)
            
            for specialty in hosp_data["specialties"]:
                spec = HospitalSpecialty(hospital_id=hosp_data["id"], name=specialty)
                session.add(spec)
            
            for equipment in hosp_data["equipments"]:
                equip = HospitalEquipment(hospital_id=hosp_data["id"], name=equipment)
                session.add(equip)
            
            for condition in hosp_data["conditions"]:
                cond = HospitalCondition(hospital_id=hosp_data["id"], icd10=condition)
                session.add(cond)
        
        await session.commit()
        print(f"Seeded {len(hospitals_data)} hospitals with their data")

if __name__ == "__main__":
    asyncio.run(seed_hospitals())