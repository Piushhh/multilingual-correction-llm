from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class CorrectionRequest(BaseModel):
    text: str = Field(..., description="The input text to be corrected.")
    language: str = Field("en", description="Language of the input text.")
    domain: str = Field("general", description="Domain of the text.")
    mock_mode: bool = Field(False, description="Use mock mode for inference safety.")
    
class BatchCorrectionRequest(BaseModel):
    requests: List[CorrectionRequest]

class ChangeItem(BaseModel):
    original: str
    corrected: str
    category: str
    position: Optional[Dict[str, int]] = None
    confidence: float = 1.0

class CorrectionResponse(BaseModel):
    corrected_text: str
    changes: List[ChangeItem]
    metadata: Dict[str, Any]

class BatchCorrectionResponse(BaseModel):
    responses: List[CorrectionResponse]
