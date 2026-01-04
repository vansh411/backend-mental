const express = require("express");
const cors = require("cors");
require("dotenv").config();
const fetch = global.fetch || require("node-fetch");

const app = express();
app.use(cors());
app.use(express.json());

/* ------------------ ML PREDICTION ------------------ */
app.post("/predict", async (req, res) => {
  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });

    if (!response.ok) {
      const txt = await response.text();
      throw new Error(`Flask server error: ${response.status} ${txt}`);
    }

    const data = await response.json();

    // Ensure we only use the first label for treatment & condition info
    const condition = (data.labels && data.labels.length > 0) ? data.labels[0] : null;
    data.condition = condition; // attach to response

    res.json(data);
  } catch (error) {
    console.error("Prediction error:", error);
    res.status(500).json({ error: "Prediction failed. Server might be offline." });
  }
});

/* ------------------ TREATMENT PLAN API ------------------ */
app.post("/treatment-plan", (req, res) => {
  const { condition } = req.body || {};
  const key = typeof condition === "string" ? condition.trim() : "";

  const plans = {
    Depression: [
      "Daily 20-minute walk (morning light exposure)",
      "Start gratitude journaling (3 items daily)",
      "Cognitive Behavioral Therapy (CBT) exercises — 2x weekly",
      "Reach out to a friend daily — small social goals",
      "Sleep routine: consistent wake/sleep times",
    ],
    Anxiety: [
      "Practice 4-7-8 breathing and box breathing daily",
      "10-minute mindfulness practice each morning",
      "Reduce caffeine and heavy stimulants",
      "Grounding (5-4-3-2-1) when anxious",
      "Consider weekly therapy (CBT) or skill-building",
    ],
    ADHD: [
      "Use Pomodoro (25/5) and visual timers",
      "Break tasks into 10–15 minute chunks",
      "Daily brief memory/executive function games",
      "Weekly planning & consistent routines",
      "Consider behavioral therapy and coaching",
    ],
    OCD: [
      "Exposure & Response Prevention (ERP) practice with guidance",
      "Track intrusive thought triggers and response patterns",
      "Delay/replace ritual behaviours gradually",
      "Relaxation and breathing practice daily",
    ],
    PTSD: [
      "Grounding techniques: name 5 things (senses)",
      "Trauma-informed breathing routines",
      "Nighttime journaling for intrusive memories",
      "Mindfulness-based stress reduction (MBSR)",
      "Consider trauma-focused therapy (TF-CBT/EMDR) referral",
    ],
    Aspergers: [
      "Social communication role-play sessions",
      "Routine building & explicit social scripts",
      "Emotional recognition exercises",
      "Join autism-friendly support groups",
    ],
    "No disorder detected": [
      "Maintain healthy lifestyle: sleep, movement, diet",
      "Daily micro-habits (5–10 min) for mental wellbeing",
      "Seek professional help if symptoms persist",
    ],
    default: [
      "Maintain healthy lifestyle: sleep, movement, diet",
      "Daily micro-habits (5–10 min) for mental wellbeing",
      "Seek professional help if symptoms persist",
    ],
  };

  res.json({ plan: plans[key] || plans.default });
});

/* ------------------ CONDITION INFORMATION ------------------ */
app.post("/condition-info", (req, res) => {
  const { condition } = req.body || {};
  const key = typeof condition === "string" ? condition.trim() : "";

  const info = {
    Depression: {
      description: "Depression is a mood disorder that causes persistent feelings of sadness and loss of interest.",
      causes: [
        "Genetic vulnerability",
        "Chemical imbalance in the brain",
        "Prolonged stress or trauma",
        "Major life changes"
      ],
      effects: [
        "Low energy, fatigue",
        "Difficulty concentrating",
        "Sleep problems",
        "Loss of interest in daily activities"
      ],
      commonEmotions: ["Sadness", "Hopelessness", "Guilt", "Irritability"]
    },
    Anxiety: {
      description: "Anxiety disorders involve excessive fear or worry that affects daily functioning.",
      causes: [
        "Genetics and family history",
        "Brain chemistry imbalances",
        "Stressful or traumatic experiences"
      ],
      effects: [
        "Rapid heartbeat, sweating",
        "Restlessness or irritability",
        "Difficulty sleeping",
        "Avoidance of certain situations"
      ],
      commonEmotions: ["Fear", "Worry", "Tension", "Apprehension"]
    },
    ADHD: {
      description: "Attention-Deficit/Hyperactivity Disorder is characterized by inattention, hyperactivity, and impulsivity.",
      causes: [
        "Genetic factors",
        "Brain development differences",
        "Environmental factors in childhood"
      ],
      effects: [
        "Difficulty focusing",
        "Impulsive actions",
        "Restlessness",
        "Difficulty organizing tasks"
      ],
      commonEmotions: ["Frustration", "Impatience", "Restlessness"]
    },
    OCD: {
      description: "Obsessive-Compulsive Disorder causes recurring unwanted thoughts and repetitive behaviors.",
      causes: [
        "Genetic predisposition",
        "Brain chemistry differences",
        "Stressful life events"
      ],
      effects: [
        "Compulsive rituals",
        "Obsessive thoughts",
        "Anxiety when rituals are not performed"
      ],
      commonEmotions: ["Anxiety", "Fear", "Frustration"]
    },
    PTSD: {
      description: "Post-Traumatic Stress Disorder occurs after experiencing or witnessing a traumatic event.",
      causes: [
        "Exposure to traumatic events",
        "Genetic vulnerability",
        "Severe stress response"
      ],
      effects: [
        "Flashbacks",
        "Nightmares",
        "Avoidance of triggers",
        "Hypervigilance"
      ],
      commonEmotions: ["Fear", "Sadness", "Anger", "Guilt"]
    },
    Aspergers: {
      description: "Asperger’s syndrome is a developmental disorder affecting social interaction and communication.",
      causes: [
        "Genetic and neurobiological factors"
      ],
      effects: [
        "Difficulty with social cues",
        "Repetitive behaviors",
        "Restricted interests"
      ],
      commonEmotions: ["Anxiety", "Confusion in social situations"]
    },
    "No disorder detected": {
      description: "No specific condition detected or information available.",
      causes: [],
      effects: [],
      commonEmotions: []
    },
    default: {
      description: "No specific condition detected or information not available.",
      causes: [],
      effects: [],
      commonEmotions: []
    }
  };

  res.json(info[key] || info.default);
});

/* ------------------ NEARBY THERAPY CENTRES ------------------ */
app.post("/nearby-centres", async (req, res) => {
  try {
    const { lat, lng, radius } = req.body || {};
    if (!lat || !lng) {
      return res.status(400).json({ error: "lat and lng are required" });
    }

    const RADIUS = Number(radius) || 5000;
    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) return res.status(500).json({ error: "Missing Google API Key" });

    const url = `https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=${lat},${lng}&radius=${RADIUS}&keyword=therapy|mental health|counselling|psychologist|psychiatrist&type=health&key=${apiKey}`;

    const response = await fetch(url);
    if (!response.ok) throw new Error(`Google API error: ${response.status}`);

    const data = await response.json();
    const centres = (data.results || []).slice(0, 12).map(c => ({
      name: c.name,
      address: c.vicinity || c.formatted_address || "",
      rating: c.rating || null,
      user_ratings_total: c.user_ratings_total || 0,
      place_id: c.place_id,
      location: c.geometry?.location || null,
      types: c.types || []
    }));

    res.json({ centres, status: data.status || "OK" });
  } catch (err) {
    console.error("nearby-centres error:", err);
    res.status(500).json({ error: "Failed to fetch nearby centres", details: err.message });
  }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`Node.js backend running on port ${PORT}`));
