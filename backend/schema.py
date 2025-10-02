from typing import List, Dict
from pydantic import Field, BaseModel

# The schema that will be received from the frontend, with the posture analysis
class PostureFeatures(BaseModel):   
    eyeContact: Dict[str, float]
    shoulderAlignment: Dict[str, float]
    handGestures: Dict[str, float]
    headBodyAlignment: Dict[str, float]

class FluencyEvaluator(BaseModel):
    comment: str = Field(..., description="3-5 sentence feedback on fluency, including pace, fillers, pauses, flow.")
    score: int = Field(..., description="Fluency score (0-100).", ge=1, le=100)

class ContentEvaluator(BaseModel):
    strengths: List[str] = Field(..., description="Strengths in content, structure, language, and grammar.", min_length=2, max_length=5)
    improvements: List[str] = Field(..., description="Suggestions for improvement in content, structure, language, grammar.", min_length=2, max_length=5)
    structure_score: int = Field(..., description="Score (0-100) for logical organization and transitions.", ge=1, le=100)
    grammar_score: int = Field(..., description="Score (0-100) for correctness of language use.", ge=1, le=100)

class SpeechEvaluator(BaseModel):
    strengths: List[str] = Field(..., description="Strengths in clarity, delivery, and perceived confidence.", min_length=2, max_length=5)
    improvements: List[str] = Field(..., description="Suggestions for improvement in clarity, delivery, perceived confidence.", min_length=2, max_length=5)
    clarity_score: int = Field(..., description="Score (0-100) for clarity of speech.", ge=1, le=100)
    confidence_score: int = Field(..., description="Score (0-100) for perceived speaker confidence.", ge=1, le=100)

class PostureEvaluator(BaseModel):
    tips: List[str] = Field(..., description="Pointers for improving posture while speaking", min_length=3, max_length=7)
    posture_score: int = Field(..., description="Score (0-100) for proper posture maintanence while speaking", ge=1, le=100)
    
class AudioOnlyFeedback(BaseModel):
    fluency_evaluator: FluencyEvaluator
    language_evaluator: ContentEvaluator
    speech_evaluator: SpeechEvaluator

# The schema that the feedback generation model will send back
class Feedback(BaseModel):      
    fluency_evaluator: FluencyEvaluator
    language_evaluator: ContentEvaluator
    speech_evaluator: SpeechEvaluator
    posture_evaluator: PostureEvaluator
    
# The response to be sent to the frontend
class FrontendResponse(BaseModel):
    transcript: str
    overall_score: int
    speaking_rate: int
    fluency_evaluator: FluencyEvaluator
    language_evaluator: ContentEvaluator
    speech_evaluator: SpeechEvaluator
    posture_evaluator: PostureEvaluator
