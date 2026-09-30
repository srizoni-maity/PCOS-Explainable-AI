from src.core.risk_scores import RiskScorer

print(RiskScorer.PACKAGED_FOOD)

print(RiskScorer.medical_score(
    "Obesity/overweight;High cholesterol"
))

print(RiskScorer.symptom_count(
    "Weight gain;Acne or oily skin"
))