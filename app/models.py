from sqlalchemy import Column, String, Enum, Text, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from geoalchemy2 import Geography
import uuid
import enum
from .db import Base

class BookingStatus(str, enum.Enum):
    pending = "pending"
    confirmed = "confirmed"
    rejected = "rejected"

class Hospital(Base):
    __tablename__ = 'hospitals'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    address = Column(Text)
    phone = Column(String)
    location = Column(Geography(geometry_type='POINT', srid=4326), nullable=False)
    
    specialties = relationship("HospitalSpecialty", back_populates="hospital", cascade="all, delete-orphan")
    equipments = relationship("HospitalEquipment", back_populates="hospital", cascade="all, delete-orphan")
    conditions = relationship("HospitalCondition", back_populates="hospital", cascade="all, delete-orphan")

class HospitalSpecialty(Base):
    __tablename__ = 'hospital_specialty'
    
    hospital_id = Column(UUID(as_uuid=True), ForeignKey('hospitals.id'), primary_key=True)
    name = Column(String, primary_key=True)
    
    hospital = relationship("Hospital", back_populates="specialties")

class HospitalEquipment(Base):
    __tablename__ = 'hospital_equipment'
    
    hospital_id = Column(UUID(as_uuid=True), ForeignKey('hospitals.id'), primary_key=True)
    name = Column(String, primary_key=True)
    
    hospital = relationship("Hospital", back_populates="equipments")

class HospitalCondition(Base):
    __tablename__ = 'hospital_condition'
    
    hospital_id = Column(UUID(as_uuid=True), ForeignKey('hospitals.id'), primary_key=True)
    icd10 = Column(String, primary_key=True)
    
    hospital = relationship("Hospital", back_populates="conditions")

class Booking(Base):
    __tablename__ = 'bookings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hospital_id = Column(UUID(as_uuid=True), ForeignKey('hospitals.id'), nullable=False)
    condition_code = Column(String)
    patient_name = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    notes = Column(Text)
    preferred_at = Column(String)
    status = Column(Enum(BookingStatus), default=BookingStatus.pending, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class DiagnosisLog(Base):
    __tablename__ = 'diagnosis_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    model_version = Column(String, nullable=False)
    top_label = Column(String)
    icd10 = Column(String)
    severity = Column(String)
    confidence_top = Column(String)