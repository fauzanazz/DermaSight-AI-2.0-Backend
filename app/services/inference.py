from ..schemas import DiagnosisResponse, ConditionProb
from ..config import settings

PREMED = {
    ("Atopic dermatitis", "mild"): ["Moisturizer twice daily", "Avoid irritants"],
    ("Atopic dermatitis", "moderate"): ["OTC hydrocortisone 1%", "Moisturizer", "Avoid scratching"],
    ("Psoriasis", "mild"): ["Moisturizer", "Gentle cleansing"],
    ("Psoriasis", "moderate"): ["Topical corticosteroid", "Moisturizer", "Avoid triggers"],
    ("Seborrheic dermatitis", "mild"): ["Antifungal shampoo", "Gentle cleansing"],
    ("Actinic keratosis", "mild"): ["Sun protection", "Gentle skincare routine"],
}

ICD_MAP = {
    "Atopic dermatitis": "L20",
    "Psoriasis": "L40", 
    "Seborrheic dermatitis": "L21",
    "Actinic keratosis": "L57.0",
    "Contact dermatitis": "L25",
}

def run_inference(img_bytes: bytes) -> DiagnosisResponse:
    top = ConditionProb(
        label="Atopic dermatitis", 
        icd10=ICD_MAP["Atopic dermatitis"], 
        confidence=0.87
    )
    others = [
        top,
        ConditionProb(label="Psoriasis", icd10=ICD_MAP["Psoriasis"], confidence=0.08),
        ConditionProb(label="Contact dermatitis", icd10=ICD_MAP["Contact dermatitis"], confidence=0.05)
    ]
    
    severity = "moderate"
    pre = PREMED.get((top.label, severity), ["Keep area clean and dry"])
    
    return DiagnosisResponse(
        topCondition=top,
        conditions=others,
        severity=severity, 
        preMedication=pre,
        advice="Seek professional evaluation if symptoms persist or worsen within 48–72h.",
        _meta={"modelVersion": settings.model_version}
    )