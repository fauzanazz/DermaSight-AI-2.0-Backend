from pydantic import BaseModel, ConfigDict
import os

class Settings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    database_url: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://skinscan:skinscan@localhost:5432/skinscan")
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    store_images: bool = os.getenv("STORE_IMAGES", "false").lower() == "true"
    model_version: str = os.getenv("MODEL_VERSION", "skinscan-derm-v1")
    model_path: str = os.getenv("MODEL_PATH", "")
    model_url: str = os.getenv("MODEL_URL", "https://huggingface.co/fauzanazz/EfficientNet-skin-disease/resolve/main/best_efficientnet.pth")
    
    # OpenAI/VLLM Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4")
    openai_vision_model: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4-vision-preview")
    use_ai_enhancement: bool = os.getenv("USE_AI_ENHANCEMENT", "true").lower() == "true"
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

settings = Settings()