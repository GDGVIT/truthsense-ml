import os
import json
import joblib
import asyncio
from groq import AsyncClient

from utils import calculate_overall_score
from prompt import get_prompt
from audio_utils import extract_features
from schema import PostureFeatures, FrontendResponse

async def get_feedback(audio_path, fluency_model, posture_features: dict | PostureFeatures, response_schema = FrontendResponse, llm_model : str = "llama-3.3-70b-versatile"):
    """Here, posture features will be gotten from the frontend as a JSON object"""
    llm_client = AsyncClient()
    
    audio_features = await extract_features(audio_path, fluency_model, llm_client)
    
    prompt = get_prompt(audio_features, posture_features, response_schema)
    
    completion = await llm_client.chat.completions.create(
        model=llm_model,
        messages=[
        {
            "role": "user",
            "content": prompt
        }
        ],
        temperature=0.5,
        max_completion_tokens=32768,
        top_p=1,
        response_format={"type": "json_object"},
        stream=False,
        stop=None,
    )
    
    response = json.loads(completion.choices[0].message.content)      # type: ignore
    response['speaking_rate'] = int(audio_features['speaking_rate'] * 60)
    print(response)
    response['overall_score'] = calculate_overall_score(response)
    response['transcript'] = audio_features['transcript']
        
    return json.dumps(response, indent=2)