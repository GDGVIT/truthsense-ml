import os
import json
import joblib
import asyncio
from groq import AsyncClient
from pydantic import BaseModel

from utils import calculate_overall_score
from prompt import get_prompt
from audio_utils import extract_features
from schema import Feedback, FrontendResponse


async def get_feedback(audio_path, fluency_model, llm_client: AsyncClient, posture_features = None, response_schema = BaseModel, llm_model : str = "llama-3.3-70b-versatile"):
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
    response['overall_score'] = calculate_overall_score(response)
    response['transcript'] = audio_features['transcript']
        
    return json.dumps(response, indent=2)

# For testing purposes
if __name__ == "__main__":
    from load_dotenv import load_dotenv
    load_dotenv(".env.local")

    import time
    start_time = time.time()
    """
    We'll need these two global variables to be defined when a new session has started, just like the session ID
    """
    client = AsyncClient()
    fluency_model = joblib.load('./fluency/models/weights/xgboost_model.pkl')

    sample_name = "tim-urban"
    feedback = asyncio.run(get_feedback(f"./samples/{sample_name}.wav", fluency_model, client, response_schema=Feedback))
    output_file = f"./backend/outputs/{sample_name}.json"
    
    if not os.path.exists("./backend/outputs/"): os.mkdir("./backend/outputs/")
    try:
        with open(output_file, "x+") as f:
            f.write(feedback)
    except FileExistsError:
        with open(output_file, "w+") as f:
            f.write(feedback)

    print(feedback)
    print(f"Total time to generate feedback and load to file: {time.time()-start_time}")