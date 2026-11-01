
"""
Breast Cancer Status Predictor - UI layer
Needs: pip install gradio joblib scikit-learn numpy pandas
"""
import html
import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Load the pre-trained model
model = joblib.load("my_model.pkl")

def predict_status(Race, Marital_Status, T_Stage, N_Stage, _6th_Stage, differentiate, Grade, A_Stage, Estrogen_Status, Progesterone_Status, Age, Tumor_Size, Regional_Node_Examined, Reginol_Node_Positive, Survival_Months):
    # Reconstruct input features in the correct order matching the model training structure
    features = np.array([[Race, Marital_Status, T_Stage, N_Stage, _6th_Stage, differentiate, Grade, A_Stage, Estrogen_Status, Progesterone_Status, Age, Tumor_Size, Regional_Node_Examined, Reginol_Node_Positive, Survival_Months]])
    prediction = model.predict(features)
    return "Dead" if prediction[0] == 1 else "Alive"

# ---------------------------------------------------------------------------
# 1. ENCODINGS  (label shown to the user -> code your model expects)
#    These assume sklearn LabelEncoder (alphabetical).
# ---------------------------------------------------------------------------
RACE     = [("White", 2), ("Black", 0), ("Other", 1)]
MARITAL  = [("Married", 1), ("Single", 3), ("Divorced", 0), ("Widowed", 4), ("Separated", 2)]
T_STAGE  = [("T1", 0), ("T2", 1), ("T3", 2), ("T4", 3)]
N_STAGE  = [("N1", 0), ("N2", 1), ("N3", 2)]
SIXTH    = [("IIA", 0), ("IIB", 1), ("IIIA", 2), ("IIIB", 3), ("IIIC", 4)]
DIFF     = [("Moderately differentiated", 0), ("Poorly differentiated", 1),
            ("Undifferentiated", 2), ("Well differentiated", 3)]
GRADE    = [("Grade I", 1), ("Grade II", 2), ("Grade III", 3), ("Grade IV (anaplastic)", 0)]
A_STAGE  = [("Regional", 1), ("Distant", 0)]
HORMONE  = [("Positive", 1), ("Negative", 0)]


def label_of(choices, code):
    return next((l for l, c in choices if c == code), str(code))


# ---------------------------------------------------------------------------
# 2. STYLE  - hematoxylin & eosin: the two stains pathologists use on a slide
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --hema: #2E2A6B;       /* hematoxylin: nuclei */
  --hema-deep: #1D1A47;
  --eosin: #F0829F;      /* eosin: cytoplasm */
  --eosin-soft: #FCE6EC;
  --glass: #F6F2F9;      /* slide glass */
  --ink: #1B1838;
  --muted: #6E6A8C;
  --ok: #1F8A7A;
  --risk: #C23B62;
}

.gradio-container {
  background: var(--glass) !important;
  font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
  max-width: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 20px clamp(16px, 3vw, 48px) !important;
}
.gradio-container > .main, .gradio-container .main, .gradio-container .wrap, .app {
  max-width: 100% !important;
  width: 100% !important;
}
footer { display: none !important; }
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
  font-family: 'Bricolage Grotesque', sans-serif !important;
  color: var(--ink);
}

/* ---- hero: a stained tissue slide ---- */
#hero {
  position: relative; overflow: hidden;
  border-radius: 22px; padding: 44px 44px 40px; margin-bottom: 18px;
  color: #fff;
  background:
    radial-gradient(circle at 8% 22%,  #5B54C4 0 7px,  transparent 8px),
    radial-gradient(circle at 15% 70%, #4A44A0 0 11px, transparent 12px),
    radial-gradient(circle at 27% 38%, #6A63D6 0 5px,  transparent 6px),
    radial-gradient(circle at 36% 82%, #5B54C4 0 8px,  transparent 9px),
    radial-gradient(circle at 58% 18%, #4A44A0 0 9px,  transparent 10px),
    radial-gradient(circle at 71% 62%, #6A63D6 0 6px,  transparent 7px),
    radial-gradient(circle at 83% 28%, #5B54C4 0 12px, transparent 13px),
    radial-gradient(circle at 92% 78%, #4A44A0 0 7px,  transparent 8px),
    radial-gradient(ellipse at 85% 110%, rgba(240,130,159,.75), transparent 55%),
    radial-gradient(ellipse at 0% 0%, rgba(240,130,159,.28), transparent 50%),
    var(--hema-deep);
}
#hero h1 {
  color: #fff !important; font-size: clamp(30px, 5vw, 54px); font-weight: 800;
  letter-spacing: -0.03em; line-height: 1.02; margin: 0 0 12px;
}
#hero p { max-width: 560px; margin: 0; color: #DAD7F5; font-size: 16px; line-height: 1.55; }
#hero .chip {
  display: inline-block; margin-top: 18px; padding: 6px 12px; border-radius: 999px;
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22);
  font-size: 13px; color: #fff;
}

/* ---- form panels ---- */
.panel {
  background: #fff !important; border: 1px solid #E4DDF0 !important;
  border-radius: 18px !important; padding: 20px 22px !important; margin-bottom: 14px !important;
  box-shadow: 0 1px 0 rgba(46,42,107,.04);
}
.panel h3 { margin: 0 0 4px !important; font-size: 19px; font-weight: 700; }
.panel .hint { color: var(--muted); font-size: 13.5px; margin: 0 0 10px; }
.panel label span, .panel .block-title { color: var(--ink) !important; font-weight: 500 !important; }
.panel input:focus, .panel textarea:focus, .panel select:focus {
  border-color: var(--hema) !important; box-shadow: 0 0 0 3px rgba(46,42,107,.15) !important;
}

/* ---- buttons ---- */
#run-btn {
  background: linear-gradient(135deg, var(--hema) 0%, #4B3FA8 55%, var(--eosin) 140%) !important;
  color: #fff !important; border: 0 !important; border-radius: 14px !important;
  font-family: 'Bricolage Grotesque', sans-serif !important; font-weight: 700 !important;
  font-size: 18px !important; padding: 16px !important;
  transition: transform .15s ease, box-shadow .15s ease;
}
#run-btn:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(46,42,107,.35); }
#reset-btn { border-radius: 14px !important; }

/* ---- result card ---- */
#result-col { position: sticky; top: 16px; align-self: flex-start; }
.res-card {
  border-radius: 20px; padding: 28px; background: #fff; border: 1px solid #E4DDF0;
  animation: pop .45s cubic-bezier(.2,.9,.3,1.2);
}
@keyframes pop { from { transform: scale(.96); opacity: 0; } to { transform: scale(1); opacity: 1; } }
.res-card.empty { background: transparent; border: 2px dashed #CFC6E6; text-align: center; color: var(--muted); animation: none; }
.res-card.empty .dot { width: 54px; height: 54px; border-radius: 50%; margin: 0 auto 14px;
  background: radial-gradient(circle at 35% 35%, #6A63D6, var(--hema)); box-shadow: 14px 10px 0 -6px var(--eosin); }
.res-card .who { font-size: 14px; color: var(--muted); margin-bottom: 6px; }
.res-card .verdict { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
  font-size: 40px; letter-spacing: -0.03em; line-height: 1.05; margin: 2px 0 14px; }
.res-card.ok   { border-top: 6px solid var(--ok); }   .res-card.ok .verdict   { color: var(--ok); }
.res-card.risk { border-top: 6px solid var(--risk); } .res-card.risk .verdict { color: var(--risk); }
.res-card.neutral { border-top: 6px solid var(--hema); } .res-card.neutral .verdict { color: var(--hema); }
.res-card.error { border-top: 6px solid var(--risk); background: var(--eosin-soft); }
.res-card table { width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 8px; }
.res-card td { padding: 7px 0; border-bottom: 1px solid #EEE9F6; }
.res-card td:first-child { color: var(--muted); width: 55%; }
.res-card td:last-child { text-align: right; font-weight: 500; color: var(--ink); }
.res-card .fine { margin-top: 16px; font-size: 12.5px; color: var(--muted); line-height: 1.5; }

@media (prefers-reduced-motion: reduce) { .res-card, #run-btn { animation: none; transition: none; } }
"""

THEME = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="pink", neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
)

HERO = """
<div id="hero">
  <h1>Breast cancer<br>status predictor</h1>
  <p>Enter a patient's clinical details and the model estimates their survival status
     from tumor stage, receptor profile and follow-up data.</p>
  <span class="chip">Machine learning project &middot; for learning, not diagnosis</span>
</div>
"""

EMPTY = """
<div class="res-card empty">
  <div class="dot"></div>
  <b>No prediction yet</b><br>
  Fill in the details on the left and press <i>Run prediction</i>.
</div>
"""


def panel_header(title, hint):
    return gr.HTML(f"<h3>{title}</h3><p class='hint'>{hint}</p>")


# ---------------------------------------------------------------------------
# 3. LOGIC - wraps predict_status
# ---------------------------------------------------------------------------
DEFAULTS = dict(name="", pid="", age=68, race=2, marital=1, size=4, t=0, n=0, sixth=0,
                a=1, diff=1, grade=2, er=1, pr=1, examined=24, positive=1, months=60)


def error_card(msg):
    return f"<div class='res-card error'><b>Check the form</b><br>{html.escape(msg)}</div>"


def run(name, pid, age, race, marital, size, t, n, sixth, a, diff, grade,
        er, pr, examined, positive, months):
    if positive > examined:
        return error_card("Positive nodes can't be higher than nodes examined.")
    if size is None or size <= 0:
        return error_card("Tumor size must be greater than 0.")

    # exact order your model was trained on
    model_inputs = [race, marital, t, n, sixth, diff, grade, a, er, pr,
                    age, size, examined, positive, months]
    try:
        raw = str(predict_status(*model_inputs))
    except Exception as e:
        return error_card(f"Prediction failed: {e}")

    low = raw.lower()
    tone = "risk" if any(w in low for w in ("dead", "deceased", "not survive")) else \
           "ok" if any(w in low for w in ("alive", "survive")) else "neutral"

    who = html.escape(name.strip()) if name and name.strip() else "Unnamed patient"
    ref = f" &middot; ID {html.escape(pid.strip())}" if pid and pid.strip() else ""
    stamp = datetime.now().strftime("%d %b %Y, %H:%M")

    rows = [
        ("Age", f"{int(age)} years"),
        ("Tumor size", f"{size:g} mm"),
        ("Stage (T / N / 6th)", f"{label_of(T_STAGE, t)} / {label_of(N_STAGE, n)} / {label_of(SIXTH, sixth)}"),
        ("Grade", label_of(GRADE, grade)),
        ("Nodes positive / examined", f"{int(positive)} / {int(examined)}"),
        ("Estrogen / Progesterone", f"{label_of(HORMONE, er)} / {label_of(HORMONE, pr)}"),
        ("Follow-up", f"{int(months)} months"),
    ]
    table = "".join(f"<tr><td>{k}</td><td>{html.escape(v)}</td></tr>" for k, v in rows)

    return f"""
    <div class="res-card {tone}">
      <div class="who">{who}{ref} &middot; {stamp}</div>
      <div class="verdict">{html.escape(raw)}</div>
      <table>{table}</table>
      <div class="fine">Output of a statistical model trained on historical data. It is not a
      medical diagnosis and must not be used for treatment decisions.</div>
    </div>"""


# ---------------------------------------------------------------------------
# 4. LAYOUT
# ---------------------------------------------------------------------------
with gr.Blocks(title="Breast Cancer Status Predictor", css=CSS, theme=THEME) as demo:
    gr.HTML(HERO)

    with gr.Row(equal_height=False):
        # ---------------- left: the form ----------------
        with gr.Column(scale=7):
            with gr.Group(elem_classes="panel"):
                panel_header("Patient", "Name and ID only label the report. They are not sent to the model.")
                with gr.Row():
                    name = gr.Textbox(label="Full name", placeholder="e.g. Priya Sharma")
                    pid = gr.Textbox(label="Patient ID (optional)", placeholder="e.g. BC-0042")
                age = gr.Slider(18, 100, value=DEFAULTS["age"], step=1, label="Age")
                with gr.Row():
                    race = gr.Dropdown(RACE, value=DEFAULTS["race"], label="Race")
                    marital = gr.Dropdown(MARITAL, value=DEFAULTS["marital"], label="Marital status")

            with gr.Group(elem_classes="panel"):
                panel_header("Tumor", "Size and staging from the pathology report.")
                size = gr.Number(value=DEFAULTS["size"], label="Tumor size (mm)", minimum=0)
                with gr.Row():
                    t = gr.Dropdown(T_STAGE, value=DEFAULTS["t"], label="T stage")
                    n = gr.Dropdown(N_STAGE, value=DEFAULTS["n"], label="N stage")
                    sixth = gr.Dropdown(SIXTH, value=DEFAULTS["sixth"], label="6th stage")
                with gr.Row():
                    a = gr.Dropdown(A_STAGE, value=DEFAULTS["a"], label="A stage")
                    diff = gr.Dropdown(DIFF, value=DEFAULTS["diff"], label="Differentiation")
                    grade = gr.Dropdown(GRADE, value=DEFAULTS["grade"], label="Grade")

            with gr.Group(elem_classes="panel"):
                panel_header("Lymph nodes", "How many were checked, and how many had cancer.")
                with gr.Row():
                    examined = gr.Number(value=DEFAULTS["examined"], label="Regional nodes examined", precision=0, minimum=0)
                    positive = gr.Number(value=DEFAULTS["positive"], label="Regional nodes positive", precision=0, minimum=0)

            with gr.Group(elem_classes="panel"):
                panel_header("Hormone receptors", "Receptor status from the biopsy.")
                with gr.Row():
                    er = gr.Radio(HORMONE, value=DEFAULTS["er"], label="Estrogen")
                    pr = gr.Radio(HORMONE, value=DEFAULTS["pr"], label="Progesterone")

            with gr.Group(elem_classes="panel"):
                panel_header("Follow-up", "Months since diagnosis.")
                months = gr.Slider(1, 120, value=DEFAULTS["months"], step=1, label="Survival months")

        # ---------------- right: sticky result ----------------
        with gr.Column(scale=5, elem_id="result-col"):
            result = gr.HTML(EMPTY)
            run_btn = gr.Button("Run prediction", elem_id="run-btn", variant="primary")
            reset_btn = gr.Button("Reset form", elem_id="reset-btn")

    # order must match run()'s signature
    fields = [name, pid, age, race, marital, size, t, n, sixth, a, diff, grade,
              er, pr, examined, positive, months]

    run_btn.click(run, inputs=fields, outputs=result)
    reset_btn.click( 
        lambda: [DEFAULTS[k] for k in ("name", "pid", "age", "race", "marital", "size", "t", "n",
                                       "sixth", "a", "diff", "grade", "er", "pr", "examined",
                                       "positive", "months")] + [EMPTY],
        outputs=fields + [result],
    )

# Explicitly listen on 0.0.0.0 and use the PORT environment variable (defaulting to 10000)
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000)),
    share=True
)
