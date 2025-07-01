WEIGHTS = {
    "clarity_score": 15,
    "confidence_score": 15,
    "fluency_score": 20,
    "structure_score": 25,
    "grammar_score": 25
}

def calculate_overall_score(result):
    mapping = {
        "fluency_score": result["fluency_coach"]["fluency_score"],
        "structure_score": result["language_coach"]["structure_score"],
        "grammar_score": result["language_coach"]["grammar_score"],
        "clarity_score": result["speech_evaluator"]["clarity_score"],
        "confidence_score": result["speech_evaluator"]["confidence_score"]
    }
    
    weighted_sum = sum(mapping[k] * WEIGHTS[k] for k in WEIGHTS)
    total_weight = sum(WEIGHTS.values())
    return int(weighted_sum / total_weight)
        