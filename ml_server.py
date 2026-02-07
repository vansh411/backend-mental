"""
Lightweight Rule-Based Mental Health Assessment Server
Use this if DistilBERT model won't load due to memory issues
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Keywords mapping for each condition
CONDITION_KEYWORDS = {
    "Depression": ["sad", "hopeless", "empty", "interest", "pleasure", "tired", "energy", "sleep", "worthless", "guilt"],
    "Anxiety": ["nervous", "anxious", "edge", "worry", "panic", "restless", "tense", "fear", "relax"],
    "ADHD": ["focus", "attention", "distracted", "concentrate", "organize", "forget", "hyperactive", "impulsive"],
    "OCD": ["repetitive", "thoughts", "compelled", "ritual", "intrusive", "control"],
    "PTSD": ["flashback", "trauma", "nightmare", "intrusive", "memories", "trigger", "upset", "avoid"],
    "Bipolar": ["energy", "manic", "high", "mood swings", "elevated"],
    "Schizophrenia": ["hear", "see", "hallucination", "voices", "paranoid"],
    "Sleep Disorder": ["sleep", "insomnia", "tired", "fatigue", "rest"]
}

def analyze_answers(questions, answers):
    """
    Rule-based analysis of questionnaire responses
    Returns: (condition, confidence, severity)
    """
    
    # Count yes answers
    yes_count = sum(1 for a in answers if a == "yes")
    total = len(answers)
    symptom_ratio = yes_count / total if total > 0 else 0
    
    # If very few symptoms, return healthy
    if symptom_ratio < 0.25:
        return "No disorder detected", 0.85, "Minimal symptoms"
    
    # Score each condition based on keyword matches
    condition_scores = {condition: 0 for condition in CONDITION_KEYWORDS.keys()}
    
    for question, answer in zip(questions, answers):
        if answer == "yes":
            question_lower = question.lower()
            
            # Check which conditions this question relates to
            for condition, keywords in CONDITION_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in question_lower:
                        condition_scores[condition] += 1
                        break  # Only count once per question
    
    # Find top conditions
    sorted_conditions = sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)
    
    top_condition, top_score = sorted_conditions[0]
    
    # If no clear match, return uncertain
    if top_score == 0:
        return "No disorder detected", 0.5, "Uncertain - general stress possible"
    
    # Calculate confidence based on score
    # Higher score relative to total = higher confidence
    confidence = min(0.95, (top_score / yes_count) * 0.9 + 0.3) if yes_count > 0 else 0.5
    
    # Determine severity
    if symptom_ratio < 0.4:
        severity = "Mild signs detected"
    elif symptom_ratio < 0.6:
        severity = "Moderate symptoms present"
    else:
        severity = "Significant symptoms detected"
    
    logger.info(f"Condition scores: {sorted_conditions[:3]}")
    logger.info(f"Selected: {top_condition} (score: {top_score}, confidence: {confidence:.2f})")
    
    return top_condition, confidence, severity


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        answers = data.get("answers", [])
        questions = data.get("questions", [])
        noSymptoms = data.get("noSymptoms", False)

        logger.info(f"Received request: {len(questions)} questions, {len(answers)} answers")

        # Handle no symptoms case
        if noSymptoms:
            return jsonify({
                "verdict": "You seem to be doing well! No significant mental health concerns detected.",
                "labels": ["No disorder detected"],
                "severity": "No symptoms detected",
                "confidence": 1.0
            })

        # Analyze using rule-based system
        condition, confidence, severity = analyze_answers(questions, answers)
        
        # Create verdict message
        if condition == "No disorder detected":
            if "Uncertain" in severity:
                verdict = f"No clear condition detected. {severity}. If you're concerned, please consult a professional."
            else:
                verdict = "You seem to be doing well! Only minor concerns detected. Consider self-care practices."
        else:
            verdict = f"{severity}. Based on your responses, you may be experiencing symptoms related to {condition}. "
            
            if confidence > 0.7:
                verdict += "Consider speaking with a mental health professional for proper evaluation."
            elif confidence > 0.5:
                verdict += "This is a preliminary assessment. Professional consultation is recommended."
            else:
                verdict += "Results are uncertain. If you're concerned, please consult a professional."

        return jsonify({
            "verdict": verdict,
            "labels": [condition],
            "severity": severity,
            "confidence": round(confidence, 2),
            "note": "Using rule-based assessment (lightweight mode)"
        })
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Lightweight rule-based server running",
        "mode": "rule-based",
        "conditions": list(CONDITION_KEYWORDS.keys())
    })


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("LIGHTWEIGHT RULE-BASED SERVER STARTING")
    logger.info("="*60)
    logger.info("This server uses keyword matching instead of ML")
    logger.info("It requires minimal memory and loads instantly")
    logger.info("="*60)
    
    app.run(port=5000, debug=True, host='0.0.0.0')
