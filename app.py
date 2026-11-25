"""Streamlit app.  Run:  streamlit run app.py"""
import io
import pandas as pd
import streamlit as st
from src import config as C
from src.predict import analyze, get_model
from src.sentiment import using_transformer

st.set_page_config(page_title="Complaint Routing System", page_icon="📨", layout="wide")
st.title("📨 Customer Complaint Routing System")
st.caption("Predicts department, sentiment severity and priority for a customer complaint.")

try:
    get_model()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

BADGE = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}
tab1, tab2, tab3 = st.tabs(["Single complaint", "Batch (CSV upload)", "Dashboard"])

# ---------------- Tab 1 ----------------
with tab1:
    examples = {
        "(write your own)": "",
        "Debt collector": "A debt collector keeps calling me at work and threatening legal action over a debt that is not mine. I want this to stop immediately.",
        "Credit report": "There is an account on my credit report that I never opened. I disputed it twice and the bureau has not removed it.",
        "Bank account": "My bank froze my checking account without notice and I cannot pay my rent. Customer service refuses to help.",
    }
    pick = st.selectbox("Try an example", list(examples))
    text = st.text_area("Complaint text", value=examples[pick], height=170)
    use_transformer_sentiment = st.checkbox(
        "Use DistilBERT sentiment (slower; may download the model)", value=False
    )
    if st.button("Analyze", type="primary") and text.strip():
        r = analyze(text, use_transformer=use_transformer_sentiment)
        c1, c2, c3 = st.columns(3)
        c1.metric("Department", r["department"]); c1.progress(min(r["confidence"], 1.0), text=f"confidence {r['confidence']:.0%}")
        c2.metric("Sentiment", r["severity"]); c2.caption(f"{r['sentiment']} | negativity {r['negativity']:.2f}")
        c3.metric("Priority", BADGE[r["priority"]]); c3.caption(f"score {r['priority_score']} | keywords: {', '.join(r['keywords']) or 'none'}")
        st.subheader("Top-3 department probabilities")
        st.bar_chart(pd.DataFrame(r["top3"], columns=["department", "probability"]).set_index("department"))
        if use_transformer_sentiment and not using_transformer():
            st.warning("Transformers not installed: sentiment is using a simple lexicon fallback. `pip install transformers torch`.")

# ---------------- Tab 2 ----------------
with tab2:
    up = st.file_uploader("Upload a CSV with a complaint text column", type="csv")
    if up:
        df = pd.read_csv(up)
        col = st.selectbox("Which column has the complaint text?", df.columns)
        limit = st.slider("Rows to process", 10, min(len(df), 2000), min(len(df), 200))
        use_sent = st.checkbox("Include transformer sentiment (slower)", value=False)
        if st.button("Run batch prediction"):
            out = df.head(limit).copy()
            rs = [analyze(str(t), with_sentiment=use_sent) for t in out[col]]
            out["pred_department"] = [r["department"] for r in rs]
            out["confidence"] = [round(r["confidence"], 3) for r in rs]
            out["priority"] = [r["priority"] for r in rs]
            if use_sent: out["severity"] = [r["severity"] for r in rs]
            st.dataframe(out, use_container_width=True)
            st.bar_chart(out["pred_department"].value_counts())
            st.download_button("Download results", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")

# ---------------- Tab 3 ----------------
with tab3:
    st.subheader("Model comparison")
    f = C.RES_DIR / "final_comparison.csv"; f = f if f.exists() else C.RES_DIR / "baseline_results.csv"
    if f.exists(): st.dataframe(pd.read_csv(f), use_container_width=True)
    imgs = [("EDA: class distribution", "eda_01_class_distribution.png"), ("EDA: complaint length", "eda_02_length.png"),
            ("Confusion matrix (TF-IDF + LogReg)", "cm_TF-IDF_LogReg.png"), ("Baseline comparison", "model_comparison_baselines.png"),
            ("Sentiment distribution", "sentiment_distribution.png"), ("Priority by department", "priority_by_department.png")]
    cols = st.columns(2)
    for i, (cap, fn) in enumerate(imgs):
        p = C.FIG_DIR / fn
        if p.exists(): cols[i % 2].image(str(p), caption=cap)
    for title, fn in [("Top words per department", "top_words_per_department.csv"), ("LDA topics", "lda_topics.csv"), ("Most confused pairs", "confused_pairs.csv")]:
        p = C.RES_DIR / fn
        if p.exists():
            st.subheader(title); st.dataframe(pd.read_csv(p), use_container_width=True)
