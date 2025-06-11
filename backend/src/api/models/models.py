from pydantic import BaseModel
from typing import List, Optional

class PredictionResult(BaseModel):
    """Model for a single image prediction result"""
    filename: str
    prediction: int
    prediction_label: str
    confidence: float
    processing_time: Optional[float] = None

class MultiPredictionResult(BaseModel):
    """Model for batch prediction results"""
    predictions: List[PredictionResult]
    total: int
    errors: int