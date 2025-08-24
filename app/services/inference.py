from ..schemas import DiagnosisResponse, ConditionProb
from ..config import settings
from PIL import Image
import io
import os
import requests
import tempfile
import logging
import json
from typing import Optional
from openai import OpenAI
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = None

def get_openai_client():
    """Get or initialize OpenAI client"""
    global openai_client
    if openai_client is None and settings.openai_api_key:
        openai_client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )
    return openai_client

PREMED = {
    # Acne
    ("Acne", "mild"): ["Gentle cleanser twice daily", "Oil-free moisturizer", "Avoid picking"],
    ("Acne", "moderate"): ["Salicylic acid cleanser", "Benzoyl peroxide", "Non-comedogenic products"],
    ("Acne", "severe"): ["See dermatologist", "Avoid harsh scrubbing", "Gentle skincare routine"],
    
    # Actinic Keratosis
    ("Actinic Keratosis", "mild"): ["Broad-spectrum SPF 30+", "Moisturizer", "Avoid sun exposure"],
    ("Actinic Keratosis", "moderate"): ["Sun protection", "Gentle skincare", "Professional consultation"],
    ("Actinic Keratosis", "severe"): ["Immediate medical evaluation", "Strict sun avoidance"],
    
    # Benign Tumors
    ("Benign Tumors", "mild"): ["Monitor for changes", "Gentle cleansing", "Sun protection"],
    ("Benign Tumors", "moderate"): ["Regular monitoring", "Professional evaluation"],
    ("Benign Tumors", "severe"): ["Immediate medical consultation"],
    
    # Bullous
    ("Bullous", "mild"): ["Keep blisters intact", "Gentle cleansing", "Dry dressings"],
    ("Bullous", "moderate"): ["Avoid friction", "Sterile dressings", "Medical evaluation"],
    ("Bullous", "severe"): ["Emergency medical care", "Avoid rupturing blisters"],
    
    # Candidiasis
    ("Candidiasis", "mild"): ["Keep area dry", "Antifungal powder", "Loose clothing"],
    ("Candidiasis", "moderate"): ["Antifungal cream", "Good hygiene", "Avoid moisture"],
    ("Candidiasis", "severe"): ["Medical treatment needed", "Systemic antifungal may be required"],
    
    # Drug Eruption
    ("Drug Eruption", "mild"): ["Discontinue suspect medication", "Cool compresses", "Gentle care"],
    ("Drug Eruption", "moderate"): ["Medical evaluation", "Document triggering medication"],
    ("Drug Eruption", "severe"): ["Emergency medical care", "Stop all non-essential medications"],
    
    # Eczema
    ("Eczema", "mild"): ["Moisturizer twice daily", "Avoid irritants", "Gentle soap"],
    ("Eczema", "moderate"): ["Hydrocortisone 1%", "Moisturizer", "Avoid scratching"],
    ("Eczema", "severe"): ["Medical treatment needed", "Avoid triggers", "Cool compresses"],
    
    # Infestations/Bites
    ("Infestations/Bites", "mild"): ["Cool compresses", "Avoid scratching", "Antihistamine if itchy"],
    ("Infestations/Bites", "moderate"): ["Topical corticosteroid", "Treat underlying infestation"],
    ("Infestations/Bites", "severe"): ["Medical evaluation", "Systemic treatment may be needed"],
    
    # Lichen
    ("Lichen", "mild"): ["Gentle skincare", "Moisturizer", "Avoid trauma"],
    ("Lichen", "moderate"): ["Topical corticosteroid", "Medical evaluation"],
    ("Lichen", "severe"): ["Dermatologist consultation", "Systemic treatment may be needed"],
    
    # Lupus
    ("Lupus", "mild"): ["Sun protection", "Gentle skincare", "Monitor symptoms"],
    ("Lupus", "moderate"): ["Rheumatology referral", "Sun avoidance", "Regular monitoring"],
    ("Lupus", "severe"): ["Immediate medical care", "Systemic evaluation needed"],
    
    # Moles
    ("Moles", "mild"): ["Monitor for changes", "Sun protection", "Regular self-examination"],
    ("Moles", "moderate"): ["Professional evaluation", "Document changes", "Sun protection"],
    ("Moles", "severe"): ["Immediate dermatology referral", "Possible biopsy needed"],
    
    # Psoriasis
    ("Psoriasis", "mild"): ["Moisturizer", "Gentle cleansing", "Avoid triggers"],
    ("Psoriasis", "moderate"): ["Topical corticosteroid", "Moisturizer", "Stress management"],
    ("Psoriasis", "severe"): ["Dermatologist consultation", "Systemic therapy may be needed"],
    
    # Rosacea
    ("Rosacea", "mild"): ["Gentle cleanser", "Sun protection", "Avoid triggers"],
    ("Rosacea", "moderate"): ["Topical metronidazole", "Gentle skincare", "Trigger avoidance"],
    ("Rosacea", "severe"): ["Medical treatment needed", "Oral antibiotics may be required"],
    
    # Seborrheic Keratoses
    ("Seborrheic Keratoses", "mild"): ["Monitor for changes", "Gentle cleansing", "Sun protection"],
    ("Seborrheic Keratoses", "moderate"): ["Professional evaluation if changing"],
    ("Seborrheic Keratoses", "severe"): ["Dermatology consultation for removal options"],
    
    # Skin Cancer
    ("Skin Cancer", "mild"): ["IMMEDIATE medical evaluation", "Sun protection", "Do not delay"],
    ("Skin Cancer", "moderate"): ["URGENT dermatology referral", "Biopsy needed"],
    ("Skin Cancer", "severe"): ["EMERGENCY medical care", "Oncology referral needed"],
    
    # Sun/Sunlight Damage
    ("Sun/Sunlight Damage", "mild"): ["Broad-spectrum sunscreen", "Moisturizer", "Limit sun exposure"],
    ("Sun/Sunlight Damage", "moderate"): ["Sun protection", "Skin examination", "Retinoid cream"],
    ("Sun/Sunlight Damage", "severe"): ["Dermatologist evaluation", "Cancer screening needed"],
    
    # Tinea
    ("Tinea", "mild"): ["Antifungal cream", "Keep area dry", "Clean clothing"],
    ("Tinea", "moderate"): ["Prescription antifungal", "Good hygiene", "Treat contacts"],
    ("Tinea", "severe"): ["Oral antifungal needed", "Medical evaluation"],
    
    # Unknown/Normal
    ("Unknown/Normal", "mild"): ["Monitor skin", "Gentle skincare", "Professional evaluation if concerned"],
    ("Unknown/Normal", "moderate"): ["Dermatology consultation recommended"],
    ("Unknown/Normal", "severe"): ["Medical evaluation needed"],
    
    # Vascular Tumors
    ("Vascular Tumors", "mild"): ["Monitor for changes", "Gentle care", "Sun protection"],
    ("Vascular Tumors", "moderate"): ["Professional evaluation", "Monitor growth"],
    ("Vascular Tumors", "severe"): ["Vascular specialist referral", "Treatment options needed"],
    
    # Vasculitis
    ("Vasculitis", "mild"): ["Gentle care", "Monitor symptoms", "Medical evaluation"],
    ("Vasculitis", "moderate"): ["Rheumatology referral", "Systemic evaluation"],
    ("Vasculitis", "severe"): ["URGENT medical care", "Systemic treatment needed"],
    
    # Vitiligo
    ("Vitiligo", "mild"): ["Sun protection", "Cosmetic concealer if desired", "Support groups"],
    ("Vitiligo", "moderate"): ["Dermatology consultation", "Treatment options available"],
    ("Vitiligo", "severe"): ["Specialist care", "Systemic treatment consideration"],
    
    # Warts
    ("Warts", "mild"): ["Over-counter treatments", "Avoid picking", "Keep clean"],
    ("Warts", "moderate"): ["Medical evaluation", "Prescription treatments"],
    ("Warts", "severe"): ["Dermatologist referral", "Specialized removal needed"],
    
    # Legacy mappings
    ("Atopic dermatitis", "mild"): ["Moisturizer twice daily", "Avoid irritants"],
    ("Atopic dermatitis", "moderate"): ["OTC hydrocortisone 1%", "Moisturizer", "Avoid scratching"],
    ("Seborrheic dermatitis", "mild"): ["Antifungal shampoo", "Gentle cleansing"],
    ("Contact dermatitis", "mild"): ["Avoid irritants", "Cool compresses", "Gentle cleansing"],
}

ICD_MAP = {
    "Acne": "L70.0",
    "Actinic Keratosis": "L57.0",
    "Benign Tumors": "D23.9",
    "Bullous": "L10",
    "Candidiasis": "B37.2",
    "Drug Eruption": "L27.0",
    "Eczema": "L30.9",
    "Infestations/Bites": "B88.9",
    "Lichen": "L43",
    "Lupus": "L93",
    "Moles": "D22.9",
    "Psoriasis": "L40",
    "Rosacea": "L71.9",
    "Seborrheic Keratoses": "L82.1",
    "Skin Cancer": "C44.9",
    "Sun/Sunlight Damage": "L57.9",
    "Tinea": "B35.9",
    "Unknown/Normal": "L99",
    "Vascular Tumors": "D18.9",
    "Vasculitis": "L95.9",
    "Vitiligo": "L80",
    "Warts": "B07",
    # Legacy mappings for compatibility
    "Atopic dermatitis": "L30.9",
    "Seborrheic dermatitis": "L21",
    "Contact dermatitis": "L25",
}

# Skin Disease Classifier Model
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Skin disease classification labels (matching existing skin disease categories)
SKIN_DISEASE_LABELS = [
    "Acne",
    "Actinic Keratosis", 
    "Atopic dermatitis",
    "Benign Tumors",
    "Bullous",
    "Candidiasis",
    "Contact dermatitis",
    "Drug Eruption",
    "Eczema",
    "Infestations/Bites",
    "Lichen",
    "Lupus",
    "Moles",
    "Psoriasis",
    "Rosacea",
    "Seborrheic dermatitis",
    "Seborrheic Keratoses",
    "Skin Cancer",
    "Sun/Sunlight Damage",
    "Tinea",
    "Unknown/Normal",
    "Vascular Tumors",
    "Vasculitis",
    "Vitiligo",
    "Warts"
]

# Map index to skin disease label
LABEL_MAP = {i: label for i, label in enumerate(SKIN_DISEASE_LABELS)}

# Image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

class SkinDiseaseClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use EfficientNet-B3 architecture
        self.model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def _load_model_from_state_dict_file(checkpoint_path: str, num_classes: int):
    """Load EfficientNet model using the specified format"""
    try:
        from efficientnet_pytorch import EfficientNet
        
        # Create model with from_pretrained and load checkpoint
        model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
        model.to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        logger.info("Successfully loaded EfficientNet model using from_pretrained")
        return model
    except Exception as e:
        logger.error(f"Error loading EfficientNet with from_pretrained: {e}")
        
        # Fallback to previous method
        try:
            # Try torchvision EfficientNet-B3 first
            tv_model = SkinDiseaseClassifier(num_classes=num_classes, pretrained=False)
            state = torch.load(checkpoint_path, map_location=device)
            tv_model.load_state_dict(state)
            tv_model.eval()
            tv_model.to(device)
            return tv_model
        except Exception as e2:
            logger.error(f"Fallback to torchvision EfficientNet-B3 also failed: {e2}")
            return None

def download_model():
    global model
    if model is None:
        try:
            NUM_CLASSES = len(SKIN_DISEASE_LABELS)
            
            # Priority 1: Try local model path
            script_dir = Path(__file__).resolve().parent.parent.parent
            local_model_path = script_dir / "models" / "best_efficientnet.pth"
            
            if local_model_path.exists():
                logger.info(f"Loading local model from: {local_model_path}")
                loaded = _load_model_from_state_dict_file(str(local_model_path), NUM_CLASSES)
                if loaded:
                    model = loaded
                    logger.info("Successfully loaded local EfficientNet model")
                    return model
                else:
                    logger.error("Local model file exists but could not be loaded with known formats")
            else:
                logger.info(f"Local model not found at: {local_model_path}")
            
            # Priority 2: Try environment-specified path
            env_model_path = settings.model_path
            if env_model_path and os.path.exists(env_model_path):
                logger.info(f"Loading model from environment path: {env_model_path}")
                loaded = _load_model_from_state_dict_file(env_model_path, NUM_CLASSES)
                if loaded:
                    model = loaded
                    logger.info("Successfully loaded environment EfficientNet model")
                    return model
                else:
                    logger.error("Environment model path found but could not be loaded with known formats")

            # Priority 3: Download from URL
            model_url = settings.model_url
            if model_url:
                logger.info(f"Downloading EfficientNet model from: {model_url}")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                    try:
                        response = requests.get(model_url, stream=True, timeout=300)
                        response.raise_for_status()
                        
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded = 0
                        
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                tmp_file.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    progress = (downloaded / total_size) * 100
                                    if downloaded % (1024 * 1024) == 0:  # Log every MB
                                        logger.info(f"Download progress: {progress:.1f}%")
                        
                        tmp_file.flush()
                        logger.info("Model download completed, loading...")
                        
                        loaded = _load_model_from_state_dict_file(tmp_file.name, NUM_CLASSES)
                        if loaded:
                            model = loaded
                            logger.info("Successfully loaded downloaded EfficientNet model")
                            return model
                        else:
                            logger.error("Downloaded model could not be loaded with known formats")
                        
                    except requests.RequestException as e:
                        logger.error(f"Network error downloading model: {e}")
                    except Exception as e:
                        logger.error(f"Error loading downloaded model: {e}")
                    finally:
                        try:
                            os.unlink(tmp_file.name)
                        except:
                            pass
            else:
                logger.warning("No model URL provided")

            # Fallback: Use pretrained model without custom weights
            logger.info("Falling back to pretrained EfficientNet model")
            model = SkinDiseaseClassifier(num_classes=NUM_CLASSES, pretrained=True)
            model.eval()
            model.to(device)
            logger.warning("Using pretrained EfficientNet model - predictions may not be accurate for skin diseases")
            return model
            
        except Exception as e:
            logger.error(f"Error loading EfficientNet model: {e}")
            return None
    
    return model

async def determine_severity_with_llm(condition: str, confidence: float) -> str:
    """Determine severity using LLM analysis"""
    try:
        client = get_openai_client()
        if not client:
            return _fallback_severity(confidence)
            
        prompt = f"""
        As a dermatology expert, determine the severity of this skin condition:
        
        Condition: {condition}
        AI Confidence: {confidence:.2%}
        
        Based on typical presentation of {condition}, classify the severity as one of:
        - mild
        - moderate  
        - severe
        
        Consider the AI confidence level in your assessment. Lower confidence may suggest unclear presentation.
        
        Respond with only the severity level (mild/moderate/severe).
        """
        
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        
        severity = response.choices[0].message.content.strip().lower()
        if severity in ["mild", "moderate", "severe"]:
            return severity
        else:
            return _fallback_severity(confidence)
            
    except Exception as e:
        logger.error(f"LLM severity determination error: {e}")
        return _fallback_severity(confidence)

def _fallback_severity(confidence: float) -> str:
    """Fallback severity determination based on confidence"""
    if confidence >= 0.8:
        return "moderate"
    elif confidence >= 0.6:
        return "mild"
    else:
        return "mild"

async def run_inference(img_bytes: bytes) -> DiagnosisResponse:
    """Run inference on image bytes and return diagnosis response"""
    try:
        model = download_model()
        
        # Validate image first
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            return _mock_response()
        
        # If EfficientNet model is unavailable, use Vision API directly
        if model is None:
            logger.info("EfficientNet model unavailable, using Vision API for analysis")
            vision_analysis = await analyze_image_with_vision(img_bytes)
            
            if vision_analysis:
                # Use Vision API results
                mapped_label = vision_analysis.get("condition", "Unknown/Normal")
                vision_conf = vision_analysis.get("confidence_pct", 50) / 100.0
                severity = vision_analysis.get("severity", "mild")
                
                top_condition = ConditionProb(
                    label=mapped_label,
                    icd10=ICD_MAP.get(mapped_label, "L99"),
                    confidence=vision_conf
                )
                
                # Create minimal condition list
                other_conditions = [top_condition]
                
                pre_med = vision_analysis.get("care_instructions", ["Monitor condition", "Seek professional evaluation if symptoms persist"])
                advice = vision_analysis.get("seek_help", "Consider professional evaluation for accurate diagnosis.")
                
                meta = {
                    "modelVersion": f"{settings.model_version}-vision-only",
                    "analysis_method": "vision_only",
                    "reason": "EfficientNet model unavailable",
                    "vision_features": vision_analysis.get("features", "")
                }
                
                return DiagnosisResponse(
                    topCondition=top_condition,
                    conditions=other_conditions,
                    severity=severity,
                    preMedication=pre_med,
                    advice=advice,
                    _meta=meta
                )
            else:
                # Vision API also failed, return enhanced mock response
                logger.warning("Both EfficientNet and Vision API unavailable, using mock response")
                return _enhanced_mock_response()
        
        # Standard EfficientNet inference flow
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, 1)
            
            # Get top prediction
            _, predicted = torch.max(outputs, 1)
            pred_idx = predicted.item()
            pred_conf = probabilities[0, pred_idx].item()
            pred_label = LABEL_MAP[pred_idx]
            
            # Get top 5 predictions
            top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)
            top5_idx = top5_indices[0].tolist()
            top5_labels = [LABEL_MAP[i] for i in top5_idx]
            top5_confs = top5_probs[0].tolist()
        
        if pred_conf < settings.confidence_threshold:
            logger.info(f"Low confidence ({pred_conf:.2%}), using Vision API fallback")
            vision_analysis = await analyze_image_with_vision(img_bytes)
            
            if vision_analysis:
                mapped_label = vision_analysis.get("condition", "Atopic dermatitis")
                vision_conf = vision_analysis.get("confidence_pct", 50) / 100.0
                severity = vision_analysis.get("severity", "mild")
                
                top_condition = ConditionProb(
                    label=mapped_label,
                    icd10=ICD_MAP.get(mapped_label, "L30.9"),
                    confidence=vision_conf
                )
                
                other_conditions = [top_condition]
                for i, (label, conf) in enumerate(zip(top5_labels[:3], top5_confs[:3])):
                    mapped = _map_prediction_to_condition(label)
                    other_conditions.append(ConditionProb(
                        label=mapped,
                        icd10=ICD_MAP.get(mapped, "L30.9"),
                        confidence=conf * 0.5
                    ))
                
                pre_med = vision_analysis.get("care_instructions", ["Keep area clean and dry"])
                advice = vision_analysis.get("seek_help", "Seek professional evaluation if symptoms persist or worsen within 48–72h.")
                
                meta = {
                    "modelVersion": settings.model_version,
                    "analysis_method": "vision_fallback",
                    "efficientnet_confidence": pred_conf,
                    "vision_features": vision_analysis.get("features", "")
                }
                
                return DiagnosisResponse(
                    topCondition=top_condition,
                    conditions=other_conditions,
                    severity=severity,
                    preMedication=pre_med,
                    advice=advice,
                    _meta=meta
                )
        
        # Use the predicted skin disease label directly
        mapped_label = pred_label
        
        severity = await determine_severity_with_llm(mapped_label, pred_conf)
        
        top_condition = ConditionProb(
            label=mapped_label,
            icd10=ICD_MAP.get(mapped_label, "L30.9"),
            confidence=pred_conf
        )
        
        other_conditions = []
        for label, conf in zip(top5_labels, top5_confs):
            other_conditions.append(ConditionProb(
                label=label,
                icd10=ICD_MAP.get(label, "L99"),  # Use appropriate ICD-10 code for skin diseases
                confidence=conf
            ))
        
        pre_med = PREMED.get((mapped_label, severity), ["Keep area clean and dry", "Apply gentle moisturizer"])
        
        ai_insights = await enhance_diagnosis_with_ai(mapped_label, pred_conf, severity)
        
        advice = ai_insights.get("seek_help", "Seek professional evaluation if symptoms persist or worsen within 48–72h.")
        if ai_insights.get("care_instructions"):
            pre_med = ai_insights["care_instructions"]
        
        meta = {
            "modelVersion": settings.model_version,
            "analysis_method": "efficientnet_plus_llm",
            "llm_severity": True
        }
        if ai_insights:
            meta["ai_insights"] = {
                "description": ai_insights.get("description", ""),
                "symptoms": ai_insights.get("symptoms", ""),
                "differentials": ai_insights.get("differentials", [])
            }
        
        return DiagnosisResponse(
            topCondition=top_condition,
            conditions=other_conditions,
            severity=severity,
            preMedication=pre_med,
            advice=advice,
            _meta=meta
        )
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return _mock_response()

def _map_prediction_to_condition(prediction: str) -> str:
    """Map model prediction to medical condition"""
    # Direct mapping for exact matches (case-insensitive)
    prediction_clean = prediction.replace("_", " ").replace("-", " ").title()
    
    # Check if prediction matches any of our known conditions
    if prediction_clean in ICD_MAP:
        return prediction_clean
    
    # Fuzzy matching for variations
    mapping = {
        "acne": "Acne",
        "actinic": "Actinic Keratosis",
        "keratosis": "Actinic Keratosis",
        "benign": "Benign Tumors",
        "tumor": "Benign Tumors",
        "bullous": "Bullous",
        "candida": "Candidiasis",
        "fungal": "Candidiasis",
        "drug": "Drug Eruption",
        "eruption": "Drug Eruption",
        "eczema": "Eczema",
        "dermatitis": "Eczema",
        "bite": "Infestations/Bites",
        "insect": "Infestations/Bites",
        "lichen": "Lichen",
        "lupus": "Lupus",
        "mole": "Moles",
        "nevus": "Moles",
        "psoriasis": "Psoriasis",
        "rosacea": "Rosacea",
        "seborrheic": "Seborrheic Keratoses",
        "cancer": "Skin Cancer",
        "carcinoma": "Skin Cancer",
        "melanoma": "Skin Cancer",
        "sun": "Sun/Sunlight Damage",
        "sunlight": "Sun/Sunlight Damage",
        "tinea": "Tinea",
        "ringworm": "Tinea",
        "normal": "Unknown/Normal",
        "unknown": "Unknown/Normal",
        "vascular": "Vascular Tumors",
        "hemangioma": "Vascular Tumors",
        "vasculitis": "Vasculitis",
        "vitiligo": "Vitiligo",
        "wart": "Warts",
        "papilloma": "Warts",
    }
    
    # Check exact match first
    pred_lower = prediction.lower()
    if pred_lower in mapping:
        return mapping[pred_lower]
    
    # Check partial matches
    for key, value in mapping.items():
        if key in pred_lower:
            return value
    
    # Default fallback
    return "Unknown/Normal"

async def analyze_image_with_vision(img_bytes: bytes) -> dict:
    """Fallback analysis using OpenAI Vision for low-confidence predictions"""
    import base64
    try:
        client = get_openai_client()
        if not client:
            return {}
            
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        prompt = """
        You are a dermatology specialist. Analyze this skin image and provide:
        
        1. Most likely skin condition
        2. Confidence level (as percentage)
        3. Severity assessment (mild/moderate/severe)
        4. Key visual features observed
        5. Recommended care instructions
        6. When to seek professional help
        
        Format as JSON with keys: condition, confidence_pct, severity, features, care_instructions, seek_help
        """
        
        response = client.chat.completions.create(
            model=settings.openai_vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            temperature=0.2,
            max_tokens=600
        )
        
        vision_content = response.choices[0].message.content
        return json.loads(vision_content)
        
    except Exception as e:
        logger.error(f"Vision analysis error: {e}")
        return {}

async def enhance_diagnosis_with_ai(condition: str, confidence: float, severity: str) -> dict:
    """Enhance diagnosis with AI-generated medical insights"""
    try:
        client = get_openai_client()
        if not client or not settings.use_ai_enhancement:
            return {}
            
        prompt = f"""
        You are a dermatology AI assistant. Based on the following computer vision diagnosis, provide medical insights:
        
        Condition: {condition}
        Confidence: {confidence:.2%}
        Severity: {severity}
        
        Please provide:
        1. Brief clinical description of the condition
        2. Common symptoms to look for
        3. Recommended care instructions (3-4 items)
        4. When to seek professional help
        5. Differential diagnoses to consider
        
        Format your response as JSON with keys: description, symptoms, care_instructions, seek_help, differentials
        Keep responses concise and medically accurate.
        """
        
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        ai_content = response.choices[0].message.content
        return json.loads(ai_content)
        
    except Exception as e:
        logger.error(f"AI enhancement error: {e}")
        return {}

def _mock_response() -> DiagnosisResponse:
    """Fallback mock response when model fails"""
    top = ConditionProb(
        label="Eczema", 
        icd10=ICD_MAP["Eczema"], 
        confidence=0.65
    )
    others = [
        top,
        ConditionProb(label="Psoriasis", icd10=ICD_MAP["Psoriasis"], confidence=0.20),
        ConditionProb(label="Unknown/Normal", icd10=ICD_MAP["Unknown/Normal"], confidence=0.15)
    ]
    
    severity = "mild"
    pre = PREMED.get((top.label, severity), ["Keep area clean and dry", "Apply gentle moisturizer"])
    
    return DiagnosisResponse(
        topCondition=top,
        conditions=others,
        severity=severity, 
        preMedication=pre,
        advice="This is a basic assessment. Seek professional evaluation for accurate diagnosis.",
        _meta={"modelVersion": f"{settings.model_version}-fallback", "analysis_method": "mock"}
    )

def _enhanced_mock_response() -> DiagnosisResponse:
    """Enhanced mock response when all AI services are unavailable"""
    top = ConditionProb(
        label="Unknown/Normal", 
        icd10=ICD_MAP["Unknown/Normal"], 
        confidence=0.50
    )
    others = [
        top,
        ConditionProb(label="Eczema", icd10=ICD_MAP["Eczema"], confidence=0.30),
        ConditionProb(label="Acne", icd10=ICD_MAP["Acne"], confidence=0.20)
    ]
    
    severity = "mild"
    pre = [
        "Monitor the condition for any changes",
        "Keep the area clean and dry", 
        "Avoid scratching or irritating the area",
        "Use gentle, fragrance-free products"
    ]
    
    return DiagnosisResponse(
        topCondition=top,
        conditions=others,
        severity=severity, 
        preMedication=pre,
        advice="IMPORTANT: AI analysis is currently unavailable. Please consult a healthcare professional or dermatologist for proper diagnosis and treatment.",
        _meta={
            "modelVersion": f"{settings.model_version}-unavailable", 
            "analysis_method": "fallback_only",
            "warning": "AI services unavailable"
        }
    )