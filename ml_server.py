"""
Ollama-Based Mental Health Assessment Server
Uses local LLM (via Ollama) for intelligent, context-aware mental health screening
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"  # You can change this to any model you have installed

# Backup keyword system (if Ollama fails)
CONDITION_KEYWORDS = {
    "Depression": ["sad", "hopeless", "empty", "interest", "pleasure", "tired", "energy", "sleep", "worthless", "guilt"],
    "Anxiety": ["nervous", "anxious", "edge", "worry", "panic", "restless", "tense", "fear", "relax"],
    "ADHD": ["focus", "attention", "distracted", "concentrate", "organize", "forget", "hyperactive", "impulsive"],
    "PTSD": ["flashback", "trauma", "nightmare", "intrusive", "memories", "trigger", "upset", "avoid"],
    "Aspergers": ["social", "cues", "routine", "literal", "sensory", "focused interests"],
}


def analyze_with_ollama(questions, answers):
    """
    Use Ollama LLM to analyze mental health assessment responses
    Returns: (condition, confidence, severity, reasoning)
    """
    
    # Build context from Q&A
    qa_pairs = []
    for q, a in zip(questions, answers):
        qa_pairs.append(f"Q: {q}\nA: {a}")
    
    qa_text = "\n\n".join(qa_pairs)
    
    # Craft a clinical prompt for the LLM
    prompt = f"""You are a mental health screening assistant. Analyze the following questionnaire responses and provide a preliminary assessment.

IMPORTANT RULES:
1. You are NOT providing a diagnosis - only a screening indication
2. Base your analysis ONLY on the conditions: Depression, Anxiety, ADHD, PTSD, Aspergers
3. If symptoms don't clearly match any condition, indicate "No disorder detected"
4. Be compassionate and professional in your language

QUESTIONNAIRE RESPONSES:
{qa_text}

Based on these responses, provide your analysis in this EXACT JSON format (no additional text):
{{
    "condition": "Depression|Anxiety|ADHD|PTSD|Aspergers|No disorder detected",
    "confidence": 0.0-1.0,
    "severity": "Minimal symptoms|Mild signs detected|Moderate symptoms present|Significant symptoms detected",
    "reasoning": "Brief explanation of why you selected this condition",
    "recommendation": "Brief supportive message and next steps"
}}

Respond with ONLY the JSON object, nothing else."""

    try:
        # Call Ollama API
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json",  # Request JSON format
                "options": {
                    "temperature": 0.3,  # Lower temperature for more consistent medical responses
                    "top_p": 0.9,
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            model_response = result.get("response", "")
            
            # Parse the JSON response
            try:
                analysis = json.loads(model_response)
                
                # Validate the response
                condition = analysis.get("condition", "No disorder detected")
                confidence = float(analysis.get("confidence", 0.5))
                severity = analysis.get("severity", "Uncertain")
                reasoning = analysis.get("reasoning", "")
                recommendation = analysis.get("recommendation", "")
                
                logger.info(f"Ollama analysis: {condition} (confidence: {confidence:.2f})")
                logger.info(f"Reasoning: {reasoning}")
                
                return condition, confidence, severity, reasoning, recommendation
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Ollama JSON response: {e}")
                logger.error(f"Raw response: {model_response}")
                raise
        else:
            logger.error(f"Ollama API error: {response.status_code}")
            raise Exception(f"Ollama returned status {response.status_code}")
            
    except Exception as e:
        logger.error(f"Ollama analysis failed: {e}")
        raise


def fallback_keyword_analysis(questions, answers):
    """
    Fallback to keyword-based analysis if Ollama fails
    """
    yes_count = sum(1 for a in answers if a == "yes")
    total = len(answers)
    symptom_ratio = yes_count / total if total > 0 else 0
    
    if symptom_ratio < 0.25:
        return "No disorder detected", 0.85, "Minimal symptoms", "Based on keyword analysis", "You appear to be doing well. Continue practicing self-care."
    
    condition_scores = {condition: 0 for condition in CONDITION_KEYWORDS.keys()}
    
    for question, answer in zip(questions, answers):
        if answer == "yes":
            question_lower = question.lower()
            for condition, keywords in CONDITION_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in question_lower:
                        condition_scores[condition] += 1
                        break
    
    sorted_conditions = sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)
    top_condition, top_score = sorted_conditions[0]
    
    if top_score == 0:
        return "No disorder detected", 0.5, "Uncertain - general stress possible", "No clear pattern detected", "If you're concerned, please consult a professional."
    
    confidence = min(0.95, (top_score / yes_count) * 0.9 + 0.3) if yes_count > 0 else 0.5
    
    if symptom_ratio < 0.4:
        severity = "Mild signs detected"
    elif symptom_ratio < 0.6:
        severity = "Moderate symptoms present"
    else:
        severity = "Significant symptoms detected"
    
    reasoning = f"Detected {top_score} indicators of {top_condition}"
    recommendation = "This is a preliminary assessment. Professional consultation is recommended."
    
    return top_condition, confidence, severity, reasoning, recommendation


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
                "confidence": 1.0,
                "method": "direct"
            })

        # Try Ollama first, fallback to keywords if it fails
        try:
            logger.info("Attempting Ollama analysis...")
            condition, confidence, severity, reasoning, recommendation = analyze_with_ollama(questions, answers)
            method = "ollama-llm"
            logger.info("✅ Ollama analysis successful")
        except Exception as e:
            logger.warning(f"Ollama failed, using keyword fallback: {e}")
            condition, confidence, severity, reasoning, recommendation = fallback_keyword_analysis(questions, answers)
            method = "keyword-fallback"
        
        # Create verdict message
        if condition == "No disorder detected":
            if "Uncertain" in severity:
                verdict = f"No clear condition detected. {severity}. {recommendation}"
            else:
                verdict = f"You seem to be doing well! {recommendation}"
        else:
            verdict = f"{severity}. Based on your responses, you may be experiencing symptoms related to {condition}. {recommendation}"
            
            if confidence > 0.7:
                verdict += " Consider speaking with a mental health professional for proper evaluation."
            elif confidence > 0.5:
                verdict += " This is a preliminary assessment. Professional consultation is recommended."
            else:
                verdict += " Results are uncertain. If you're concerned, please consult a professional."

        return jsonify({
            "verdict": verdict,
            "labels": [condition],
            "severity": severity,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
            "method": method,
            "note": "AI-powered assessment using Ollama" if method == "ollama-llm" else "Using keyword-based analysis"
        })
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Check if server and Ollama are running"""
    ollama_status = "unknown"
    ollama_models = []
    
    try:
        # Check if Ollama is available
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            ollama_status = "connected"
            models_data = response.json()
            ollama_models = [m.get("name", "unknown") for m in models_data.get("models", [])]
    except:
        ollama_status = "not available"
    
    return jsonify({
        "status": "healthy",
        "message": "Mental health assessment server running",
        "mode": "ollama-with-fallback",
        "ollama_status": ollama_status,
        "ollama_models": ollama_models,
        "current_model": MODEL_NAME,
        "conditions": list(CONDITION_KEYWORDS.keys())
    })


@app.route("/models", methods=["GET"])
def list_models():
    """List available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Failed to fetch models"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("OLLAMA-POWERED MENTAL HEALTH ASSESSMENT SERVER")
    logger.info("="*60)
    logger.info(f"Primary: Ollama LLM ({MODEL_NAME})")
    logger.info("Fallback: Keyword-based analysis")
    logger.info("="*60)
    
    # Check Ollama availability at startup
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            logger.info(f"✅ Ollama is running with {len(models)} model(s)")
            model_names = [m.get("name") for m in models]
            logger.info(f"Available models: {', '.join(model_names)}")
            
            if MODEL_NAME not in model_names and f"{MODEL_NAME}:latest" not in model_names:
                logger.warning(f"⚠️  Model '{MODEL_NAME}' not found!")
                logger.warning(f"Available models: {', '.join(model_names)}")
                logger.warning("Update MODEL_NAME in the code or pull the model:")
                logger.warning(f"  ollama pull {MODEL_NAME}")
        else:
            logger.warning("⚠️  Ollama is not responding - will use keyword fallback")
    except Exception as e:
        logger.warning(f"⚠️  Cannot connect to Ollama: {e}")
        logger.warning("Server will use keyword-based fallback")
    
    logger.info("="*60)
    
    app.run(port=5000, debug=True, host='0.0.0.0')