from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

# Load model + tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("./model")
model = DistilBertForSequenceClassification.from_pretrained("./model")
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    answers = data.get("answers", [])
    questions = data.get("questions", [])
    noSymptoms = data.get("noSymptoms", False)

    # 1. IMMEDIATE RETURN IF USER HAS NO SYMPTOMS
    if noSymptoms:
        return jsonify({
            "verdict": "Healthy / No significant issue",
            "labels": [],
            "severity": "No symptoms detected"
        })

    # 2. Convert yes/no scores
    score = sum(1 for a in answers if a == "yes")
    avg_score = score / len(answers)

    # 3. Basic severity detection
    if avg_score < 0.2:
        severity = "Mild signs, nothing serious."
    elif avg_score < 0.4:
        severity = "Moderate signs — monitor yourself."
    else:
        severity = "Possible psychological concern detected."

    # 4. Prepare text for classification
    combined_text = ". ".join([f"Q: {q} A: {a}" for q, a in zip(questions, answers)])

    encoded = tokenizer(
        combined_text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits

    # 5. Sigmoid probabilities
    probs = torch.sigmoid(logits)[0].tolist()

    # 6. Pick only the **highest probability label**
    max_index = probs.index(max(probs))
    selected_label = model.config.id2label[max_index]

    # 7. Handle no disorder
    if selected_label.lower() in ["no disorder detected", "none"]:
        selected_label = "No disorder detected"

    final_verdict = f"{severity}. Possible indicator: {selected_label}."

    return jsonify({
        "verdict": final_verdict,
        "labels": [selected_label],
        "severity": severity
    })


if __name__ == "__main__":
    app.run(port=5000)
