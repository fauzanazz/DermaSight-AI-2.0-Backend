from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from ..schemas import DiagnosisResponse
from ..services.inference import run_inference
from ..models import DiagnosisLog
from ..db import get_db
from ..config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["diagnose"])

@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(image: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    # Validate image
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported media type")
    
    img_bytes = await image.read()
    if len(img_bytes) < 1024:
        raise HTTPException(status_code=400, detail="Image too small")
    
    if len(img_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="Image too large")
    
    # Run inference (this should always work with fallbacks)
    result = await run_inference(img_bytes)
    
    # Try to log to database, but don't fail if database is unavailable
    try:
        log_entry = DiagnosisLog(
            model_version=settings.model_version,
            top_label=result.topCondition.label,
            icd10=result.topCondition.icd10,
            severity=result.severity,
            confidence_top=str(result.topCondition.confidence)
        )
        db.add(log_entry)
        await db.commit()
        logger.info(f"Logged diagnosis: {result.topCondition.label}")
    except SQLAlchemyError as e:
        logger.error(f"Database logging failed: {e}")
        # Add warning to metadata but don't fail the request
        if result._meta is None:
            result._meta = {}
        result._meta["database_logging"] = "failed"
        # Rollback the transaction
        try:
            await db.rollback()
        except:
            pass
    except Exception as e:
        logger.error(f"Unexpected error during database logging: {e}")
        if result._meta is None:
            result._meta = {}
        result._meta["database_logging"] = "error"
        try:
            await db.rollback()
        except:
            pass
    
    return result