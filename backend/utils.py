WEIGHTS = {
    "clarity_score": 15,
    "confidence_score": 15,
    "fluency_score": 15,
    "structure_score": 20,
    "grammar_score": 20,
    "posture_score": 15,
}

def calculate_overall_score(result):
    mapping = {
        "fluency_score": result["fluency_evaluator"]["fluency_score"],
        "structure_score": result["language_evaluator"]["structure_score"],
        "grammar_score": result["language_evaluator"]["grammar_score"],
        "clarity_score": result["speech_evaluator"]["clarity_score"],
        "confidence_score": result["speech_evaluator"]["confidence_score"]
    }
    
    weighted_sum = sum(mapping[k] * WEIGHTS[k] for k in WEIGHTS if k in mapping.keys())
    total_weight = sum(WEIGHTS.values())
    return int(weighted_sum / total_weight)
        