from typing import List
from pydantic import Field, BaseModel

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
    nothing: None

class AudioOnlyFeedback(BaseModel):
    fluency_evaluator: FluencyEvaluator
    language_evaluator: ContentEvaluator
    speech_evaluator: SpeechEvaluator

class Feedback(BaseModel):
    fluency_evaluator: FluencyEvaluator
    language_evaluator: ContentEvaluator
    speech_evaluator: SpeechEvaluator
    posture_evaluator: PostureEvaluator  # Placeholder for now, schema not decided yet
    
class FrontendResponse(BaseModel):
    transcript: str
    overall_score: int
    fluency_evaluator: FluencyEvaluator
    language_evaluator: ContentEvaluator
    speech_evaluator: SpeechEvaluator
    posture_evaluator: PostureEvaluator  # Placeholder for now, schema not decided yet
