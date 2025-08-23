# DermaSight AI Backend

A FastAPI backend for skin condition analysis and hospital booking system.

## Features

- 🔬 **AI Diagnosis**: Mock skin condition analysis with confidence scores
- 🏥 **Hospital Search**: Geospatial search for qualified dermatology centers  
- 📅 **Booking System**: Patient appointment scheduling
- 🗄️ **PostgreSQL + PostGIS**: Efficient geospatial queries
- 📊 **Logging**: Diagnosis analytics and audit trails

## Quick Start

1. **Prerequisites**
   ```bash
   docker --version
   docker-compose --version
   python3 --version
   ```

2. **Start the application**
   ```bash
   python start.py
   ```

3. **Access the API**
   - API Documentation: http://localhost:8000/docs  
   - Health Check: http://localhost:8000/health

## Manual Setup

If you prefer manual setup:

```bash
# Start database
docker-compose up -d db

# Wait for DB to be ready (10 seconds)
sleep 10

# Initialize database
python init_db.py

# Seed sample data  
python seed_data.py

# Start API server
docker-compose up -d api
```

## API Endpoints

### POST /api/v1/diagnose
Upload an image for skin condition analysis.

**Request**: `multipart/form-data` with `image` file
**Response**:
```json
{
  "topCondition": {"label": "Atopic dermatitis", "icd10": "L20", "confidence": 0.87},
  "conditions": [...],
  "severity": "moderate", 
  "preMedication": ["OTC hydrocortisone 1%", "Moisturizer"],
  "advice": "Seek professional evaluation if symptoms persist..."
}
```

### GET /api/v1/hospitals
Find nearby hospitals with dermatology capabilities.

**Parameters**:
- `lat`, `lng`: Location coordinates  
- `condition`: ICD-10 code filter (optional)
- `radius_km`: Search radius (default: 20km)

**Response**: Array of hospitals with distance, specialties, equipment

### POST /api/v1/bookings  
Create a new appointment booking.

**Request**:
```json
{
  "hospitalId": "uuid",
  "conditionCode": "L20",
  "patientName": "John Doe", 
  "phone": "+628123456789",
  "notes": "Urgent consultation needed",
  "preferredAt": "2024-01-15 14:00"
}
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run database migrations
alembic revision --autogenerate -m "Description"
alembic upgrade head
```

## Architecture

```
FastAPI Application
├── Diagnosis Pipeline (Mock AI model)
├── Geospatial Hospital Search (PostGIS)  
├── Booking Management (SQLAlchemy)
└── Audit Logging (PostgreSQL)
```

## Database Schema

- `hospitals`: Location, contact info, PostGIS point geometry
- `hospital_specialty`: Specializations (dermatology, etc.)
- `hospital_equipment`: Available equipment and capabilities  
- `hospital_condition`: ICD-10 conditions they can treat
- `bookings`: Patient appointment requests
- `diagnosis_logs`: AI model inference tracking

## Configuration

Environment variables:
- `DATABASE_URL`: PostgreSQL connection string
- `CORS_ORIGINS`: Comma-separated allowed origins  
- `STORE_IMAGES`: Enable image storage (default: false)
- `MODEL_VERSION`: AI model identifier

## Production Notes

- Add authentication/authorization
- Implement real ML model (ONNX/TorchScript)
- Set up monitoring (Prometheus/Grafana)
- Configure reverse proxy (Nginx)
- Enable HTTPS/SSL termination
- Add rate limiting and security headers