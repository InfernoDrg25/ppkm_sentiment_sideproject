import streamlit as st
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# CONFIG & LABEL MAPPING
# ---------------------------

st.set_page_config(
    page_title="Analisis Sentimen PPKM",
    layout="wide"
)

ID2LABEL = {0: "Positive", 1: "Neutral", 2: "Negative"}
LABEL2COLOR = {
    "Positive": "green",
    "Neutral": "gray",
    "Negative": "red"
}

MODEL_PATH = "ppkm_sentiment_pipeline.pkl"
MATRIX_PATH = "ppkm_tfidf_full.npz"
META_PATH = "ppkm_corpus_metadata.csv"


# ---------------------------
# LOADING FUNCTIONS (CACHED)
# ---------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model


@st.cache_resource
def load_corpus():
    X = sp.load_npz(MATRIX_PATH)
    meta = pd.read_csv(META_PATH)
    return X, meta


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def get_proba_dict(model, proba_array):
    """
    Map probabilities to human-readable labels:
    { "Positive": p, "Neutral": p, "Negative": p }
    """
    clf = model.named_steps["logreg"]
    classes = clf.classes_  # e.g. [0, 1, 2]
    proba_dict = {}
    for idx, c in enumerate(classes):
        proba_dict[ID2LABEL[c]] = float(proba_array[idx])
    return proba_dict


def compute_sentiment_score(proba_dict):
    """
    Simple sentiment score in range [-1, 1] approx:
    score = P(Positive) - P(Negative)
    Neutral naturally dampens both sides.
    """
    p_pos = proba_dict.get("Positive", 0.0)
    p_neg = proba_dict.get("Negative", 0.0)
    score = p_pos - p_neg  # negative -> more negative, positive -> more positive
    return score


def describe_score(score):
    """
    Turn numeric score into a simple text description.
    """
    if score <= -0.6:
        return "Sangat negatif"
    elif score <= -0.2:
        return "Cukup negatif"
    elif score < 0.2:
        return "Netral / campuran"
    elif score < 0.6:
        return "Cukup positif"
    else:
        return "Sangat positif"


def analyze_text_meta(text):
    """
    Return simple metadata: length, words, negation, emoji.
    """
    txt = text.strip()
    n_chars = len(txt)
    words = txt.split()
    n_words = len(words)

    lower = txt.lower()
    negations = ["tidak", "nggak", "gak", "ga", "tak", "bukan", "tdk", "enggak"]
    has_negation = any(neg in lower.split() for neg in negations)

    # simple check for emojis/emoticons
    emoticons = [":)", ":(", ":D", ";)", ":-)", ":'("]
    emoji_chars = ["😂", "😭", "😡", "😢", "😃", "😅", "😆", "😊", "😍", "🤔", "😷"]
    has_emoji = any(e in txt for e in emoticons + emoji_chars)

    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "has_negation": has_negation,
        "has_emoji": has_emoji,
    }


def get_top_contributing_words(model, text, predicted_class_id, top_k=5):
    """
    Use TF-IDF * Logistic Regression coefficients to estimate
    which tokens contribute most to the predicted class.
    Returns list of (token, contribution) sorted by contribution desc.
    """
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["logreg"]

    X = vectorizer.transform([text])  # 1 x n_features
    feature_names = vectorizer.get_feature_names_out()
    x = X.toarray()[0]

    # find index of predicted class in clf.classes_
    classes = clf.classes_
    class_index = list(classes).index(predicted_class_id)
    coefs = clf.coef_[class_index]  # shape: (n_features,)

    contributions = x * coefs  # element-wise
    # sort by contribution descending
    top_idx = np.argsort(contributions)[::-1]

    top_features = []
    for idx in top_idx:
        if contributions[idx] <= 0:
            # stop if we hit non-positive contribution
            continue
        token = feature_names[idx]
        weight = float(contributions[idx])
        # only keep tokens that actually appear in the text
        if token in text.lower():
            top_features.append((token, weight))
        if len(top_features) >= top_k:
            break

    return top_features


def find_similar_tweets(input_text, predicted_label_id, model, corpus_matrix, meta_df, top_k=3):
    """
    Find top-k most similar tweets (cosine similarity) with the same sentiment label.
    Returns list of dicts: {tweet, sentiment_label, similarity}
    """
    vectorizer = model.named_steps["tfidf"]
    input_vec = vectorizer.transform([input_text])

    # cosine similarity with entire corpus
    sims = cosine_similarity(input_vec, corpus_matrix)[0]  # shape: (n_samples,)

    # only keep rows with same sentiment label
    mask = meta_df["sentiment"] == predicted_label_id
    idxs = np.where(mask)[0]

    if len(idxs) == 0:
        return []

    sims_same = sims[idxs]

    # get top indices within this subset
    top_relative = np.argsort(sims_same)[::-1][:top_k]

    similar = []
    for r_idx in top_relative:
        corpus_idx = idxs[r_idx]
        row = meta_df.iloc[corpus_idx]
        sim_score = float(sims[corpus_idx])
        similar.append({
            "tweet": row["Tweet"],
            "sentiment_label": ID2LABEL[row["sentiment"]],
            "similarity": sim_score
        })

    return similar


# ---------------------------
# UI LAYOUT
# ---------------------------

st.title("🧪 Analisis Sentimen Tweet PPKM (Placebo Dataset)")
st.write(
    """
Model ini dilatih pada dataset tweet berbahasa Indonesia mengenai PPKM (2020–2022).
Masukkan sebuah kalimat/tweet, lalu aplikasi akan:
- Memprediksi **sentimen utama** (Positive, Neutral, Negative)  
- Menampilkan **probabilitas per kelas**  
- Memberikan **skor intensitas sentimen**  
- Menunjukkan **kata-kata kunci** yang berkontribusi  
- Menyajikan **3 tweet lain yang mirip** dari dataset dengan sentimen yang sama  
- Menampilkan **info sederhana tentang teks** yang Anda masukkan  
"""
)

st.markdown("---")

user_text = st.text_area(
    "Masukkan tweet / kalimat bahasa Indonesia:",
    height=120,
    placeholder="Contoh: PPKM ini bikin pusing, tapi saya mengerti kenapa perlu diterapkan."
)

analyze_button = st.button("🔍 Analisis Sentimen")

if analyze_button:
    if not user_text.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        # Load model and corpus
        with st.spinner("Memuat model dan korpus..."):
            model = load_model()
            corpus_matrix, meta_df = load_corpus()

        # ---------------------------
        # PREDICTION
        # ---------------------------
        with st.spinner("Menganalisis sentimen..."):
            proba = model.predict_proba([user_text])[0]
            pred_class_id = model.predict([user_text])[0]
            pred_label = ID2LABEL[pred_class_id]
            proba_dict = get_proba_dict(model, proba)
            score = compute_sentiment_score(proba_dict)
            score_desc = describe_score(score)
            meta = analyze_text_meta(user_text)
            top_words = get_top_contributing_words(model, user_text, pred_class_id, top_k=5)
            similar_tweets = find_similar_tweets(
                user_text, pred_class_id, model, corpus_matrix, meta_df, top_k=3
            )

        # ---------------------------
        # MAIN RESULT
        # ---------------------------
        st.markdown("### 🎯 Hasil Utama")

        col_main, col_score = st.columns([2, 1])

        with col_main:
            color = LABEL2COLOR.get(pred_label, "gray")
            st.markdown(
                f"""
                <div style="padding:1rem; border-radius:0.5rem; background-color:#f5f5f5; border-left:6px solid {color};">
                    <span style="font-size:1rem; color:#555;">Prediksi sentimen:</span><br>
                    <span style="font-size:1.6rem; font-weight:700; color:{color};">{pred_label}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(
                f"Model menilai teks ini sebagai **{pred_label}** "
                f"dengan kecenderungan: **{score_desc}**."
            )

        with col_score:
            st.markdown("**Skor Sentimen (perkiraan)**")
            st.metric(
                label="Skor (Positive − Negative)",
                value=f"{score:.2f}",
                delta=score_desc
            )

        # ---------------------------
        # PROBABILITY BREAKDOWN
        # ---------------------------
        st.markdown("### 📊 Probabilitas per Kelas")

        proba_df = pd.DataFrame(
            {
                "Sentiment": list(proba_dict.keys()),
                "Probability": list(proba_dict.values())
            }
        ).set_index("Sentiment")

        st.bar_chart(proba_df)

        # ---------------------------
        # TOP WORDS / EXPLANATION
        # ---------------------------
        st.markdown("### 🧠 Kata-kata yang Berkontribusi")

        if top_words:
            st.write("Kata/phrase berikut diperkirakan paling mendorong prediksi model:")

            for token, weight in top_words:
                st.markdown(
                    f"- **`{token}`** (kontribusi ≈ `{weight:.4f}`)"
                )
        else:
            st.write("Tidak ditemukan kata yang berkontribusi kuat (menurut model).")

        # ---------------------------
        # TEXT META INFO
        # ---------------------------
        st.markdown("### 🧾 Info Singkat tentang Teks")

        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.write(f"- Jumlah karakter: **{meta['n_chars']}**")
            st.write(f"- Jumlah kata: **{meta['n_words']}**")
        with col_meta2:
            st.write(f"- Mengandung negasi (tidak/nggak/gak/ga/tak/bukan/...): **{'Ya' if meta['has_negation'] else 'Tidak'}**")
            st.write(f"- Mengandung emoji/emotikon: **{'Ya' if meta['has_emoji'] else 'Tidak'}**")

        # ---------------------------
        # SIMILAR TWEETS
        # ---------------------------
        st.markdown("### 🔁 Tweet Lain yang Mirip (dari Dataset PPKM)")

        if similar_tweets:
            st.write(
                "Berikut beberapa tweet dari dataset yang **mirip** dengan teks Anda "
                f"dan memiliki sentimen **{pred_label}**:"
            )
            for i, item in enumerate(similar_tweets, start=1):
                st.markdown(
                    f"""
                    **{i}. ({item['sentiment_label']})**  
                    _Similarity: {item['similarity']:.3f}_  
                    > {item['tweet']}
                    """
                )
        else:
            st.write("Tidak ditemukan tweet mirip dengan sentimen yang sama di korpus.")


else:
    st.info("Masukkan sebuah teks, lalu klik **Analisis Sentimen** untuk memulai.")
