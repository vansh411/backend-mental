"""
Simplified Ollama Mental Health Server
Optimized for slower systems
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import requests
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "tinyllama"

# Simplified keywords for fallback
CONDITION_KEYWORDS = {
    "Depression": ["sad", "hopeless", "empty", "tired", "worthless", "sleep"],
    "Anxiety": ["nervous", "anxious", "worry", "panic", "restless", "fear"],
    "ADHD": ["focus", "attention", "distracted", "concentrate", "forget"],
    "PTSD": ["flashback", "trauma", "nightmare", "memories", "avoid"],
    "Aspergers": ["social", "routine", "literal", "sensory"],
}


def test_ollama_connection():
    """Quick test if Ollama is responsive"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": "Say 'OK'",
                "stream": False
            },
            timeout=10
        )
        return response.status_code == 200
    except:
        return False


def analyze_with_ollama_simple(questions, answers):
    """Simplified Ollama analysis with shorter prompt"""
    
    # Count yes/no
    yes_count = sum(1 for a in answers if a == "yes")
    
    # Build minimal context (only YES answers to save tokens)
    yes_questions = [q for q, a in zip(questions, answers) if a == "yes"]
    
    if len(yes_questions) == 0:
        return "No disorder detected", 0.9, "No symptoms", "No symptoms reported", "Keep up the good work!"
    
    # Very short prompt
    prompt = f"""You are a mental health screening assistant. Analyze these symptoms and pick ONE primary condition.

Reported symptoms:
{chr(10).join('- ' + q for q in yes_questions[:10])}

Choose the SINGLE MOST LIKELY condition from: Depression, Anxiety, ADHD, PTSD, Aspergers, or "No disorder detected"

Respond ONLY with valid JSON (no markdown, no extra text):
{{"condition":"Depression","confidence":0.85,"severity":"Moderate"}}

Rules:
- Pick ONLY ONE condition (the most prominent)
- If symptoms are mixed, choose the strongest pattern
- confidence: 0.0 to 1.0
- severity: "Mild" or "Moderate" or "Significant"

Your JSON response:"""

    try:
        logger.info(f"Sending to Ollama: {len(yes_questions)} symptoms")
        start_time = time.time()
        
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 100,  # Limit response length
                }
            },
            timeout=120  # 2 minutes max
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Ollama responded in {elapsed:.1f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            model_response = result.get("response", "")
            
            # Clean response
            model_response = model_response.strip()
            if model_response.startswith("```"):
                model_response = model_response.split("```")[1]
            if model_response.startswith("json"):
                model_response = model_response[4:].strip()
            
            logger.info(f"Raw response: {model_response[:200]}")
            
            try:
                analysis = json.loads(model_response)
                condition = analysis.get("condition", "No disorder detected")
                confidence = float(analysis.get("confidence", 0.5))
                severity = analysis.get("severity", "Uncertain")
                
                # Validate condition is one of our expected values
                valid_conditions = ["Depression", "Anxiety", "ADHD", "PTSD", "Aspergers", "No disorder detected"]
                if condition not in valid_conditions:
                    # Try to find a valid condition in the response
                    for valid in valid_conditions:
                        if valid.lower() in condition.lower():
                            condition = valid
                            break
                    else:
                        # Default to most common if still invalid
                        logger.warning(f"Invalid condition '{condition}', defaulting to No disorder detected")
                        condition = "No disorder detected"
                        confidence = 0.5
                
                # Validate severity
                valid_severities = ["Mild", "Moderate", "Significant", "Minimal", "Uncertain"]
                if severity not in valid_severities:
                    severity = "Moderate"
                
                reasoning = f"AI analysis based on {len(yes_questions)} reported symptoms"
                recommendation = "Professional evaluation recommended if symptoms persist"
                
                return condition, confidence, severity, reasoning, recommendation
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse failed: {e}")
                logger.error(f"Response was: {model_response}")
                raise
        else:
            raise Exception(f"Ollama returned {response.status_code}")
            
    except Exception as e:
        logger.error(f"Ollama failed: {e}")
        raise


def fallback_keyword_analysis(questions, answers):
    """Quick keyword-based analysis"""
    yes_count = sum(1 for a in answers if a == "yes")
    total = len(answers)
    ratio = yes_count / total if total > 0 else 0
    
    if ratio < 0.25:
        return "No disorder detected", 0.85, "Minimal", "Few symptoms", "You're doing well"
    
    scores = {c: 0 for c in CONDITION_KEYWORDS}
    
    for q, a in zip(questions, answers):
        if a == "yes":
            q_lower = q.lower()
            for condition, keywords in CONDITION_KEYWORDS.items():
                if any(kw in q_lower for kw in keywords):
                    scores[condition] += 1
    
    top_condition = max(scores, key=scores.get)
    top_score = scores[top_condition]
    
    if top_score == 0:
        return "No disorder detected", 0.5, "Uncertain", "No clear pattern", "Consult if concerned"
    
    confidence = min(0.9, (top_score / yes_count) * 0.8 + 0.4) if yes_count > 0 else 0.5
    severity = "Mild" if ratio < 0.4 else "Moderate" if ratio < 0.6 else "Significant"
    
    return top_condition, confidence, severity, f"{top_score} indicators", "Professional consultation recommended"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        answers = data.get("answers", [])
        questions = data.get("questions", [])
        noSymptoms = data.get("noSymptoms", False)

        logger.info(f"Assessment: {len(questions)} questions")

        if noSymptoms:
            return jsonify({
                "verdict": "You're doing well! No significant concerns detected.",
                "labels": ["No disorder detected"],
                "severity": "No symptoms",
                "confidence": 1.0,
                "method": "direct"
            })

        # Try Ollama, fallback to keywords
        try:
            logger.info("Trying Ollama...")
            condition, confidence, severity, reasoning, recommendation = analyze_with_ollama_simple(questions, answers)
            method = "ollama-ai"
            logger.info(f"✅ Ollama success: {condition}")
        except Exception as e:
            logger.warning(f"⚠️  Ollama failed, using keywords: {e}")
            condition, confidence, severity, reasoning, recommendation = fallback_keyword_analysis(questions, answers)
            method = "keyword-fallback"
        
        # Build verdict
        if condition == "No disorder detected":
            verdict = f"You seem to be doing well! {recommendation}"
        else:
            verdict = f"{severity} symptoms detected. You may be experiencing {condition}. {recommendation}"

        return jsonify({
            "verdict": verdict,
            "labels": [condition],
            "severity": severity,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
            "method": method
        })
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    ollama_ok = test_ollama_connection()
    
    return jsonify({
        "status": "healthy",
        "ollama_status": "connected" if ollama_ok else "timeout/not available",
        "model": MODEL_NAME,
        "fallback": "keyword analysis available"
    })


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("SIMPLIFIED OLLAMA MENTAL HEALTH SERVER")
    logger.info("="*60)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info("Testing Ollama connection...")
    
    if test_ollama_connection():
        logger.info("✅ Ollama is responding")
    else:
        logger.warning("⚠️  Ollama is slow/unavailable - will use keyword fallback")
    
    logger.info("="*60)
    
    app.run(port=5000, debug=True, host='0.0.0.0')
