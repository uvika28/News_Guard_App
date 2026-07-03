#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.sequence import pad_sequences
from newspaper import Article
from urllib.parse import urlparse
import pdfplumber
import os
import gdown

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="NewsGuard AI",
    page_icon="📰",
    layout="wide"
)

MAX_LEN = 500

# ==================================================
# DATA
# ==================================================

@st.cache_data
def load_data():

    try:
        fake = pd.read_csv("Fake.csv")
        true = pd.read_csv("True.csv")

        fake["label"] = 1
        true["label"] = 0

        df = pd.concat([fake, true], ignore_index=True)

        return df

    except Exception as e:
        st.error(f"Dataset Error: {e}")
        return pd.DataFrame()


df = load_data()

# ==================================================
# MODEL
# ==================================================

@st.cache_resource
def load_model():

    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(
        MODEL_DIR,
        "bigru_attention_model.keras"
    )

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Download model automatically if missing
    if not os.path.exists(MODEL_PATH):

        FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"

        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=False
        )

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH
        )

        with open(
            os.path.join(
                MODEL_DIR,
                "tokenizer.pkl"
            ),
            "rb"
        ) as f:

            tokenizer = pickle.load(f)

        return model, tokenizer

    except Exception as e:

        st.error(f"Model Loading Error: {e}")

        return None, None


model, tokenizer = load_model()

# ==================================================
# SOURCE CREDIBILITY
# ==================================================

SOURCE_SCORES = {

    "bbc.com": 95,
    "reuters.com": 98,
    "cnn.com": 85,
    "foxnews.com": 80,
    "nytimes.com": 96,
    "theguardian.com": 94

}

# ==================================================
# HELPERS
# ==================================================

def predict_article(text):

    if model is None or tokenizer is None:
        return "MODEL NOT LOADED", 0

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        padded,
        verbose=0
    )[0][0]

    confidence = round(
        max(prediction, 1 - prediction) * 100,
        2
    )

    label = (
        "FAKE NEWS"
        if prediction > 0.5
        else "REAL NEWS"
    )

    return label, confidence


def extract_news(url):

    try:

        article = Article(url)

        article.download()
        article.parse()

        return {

            "title": article.title,
            "authors": article.authors,
            "text": article.text,
            "date": article.publish_date

        }

    except Exception as e:

        st.error(f"Extraction Error: {e}")

        return None


def display_prediction(label, confidence):

    if label == "FAKE NEWS":

        st.error(f"🚨 {label}")

    elif label == "REAL NEWS":

        st.success(f"✅ {label}")

    else:

        st.warning(label)

    st.metric(
        "Confidence",
        f"{confidence}%"
    )

    st.progress(int(confidence))

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("📰 NewsGuard AI")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Fake News Detection",
        "News Labeling",
        "Analytics",
        "Dataset Explorer"
    ]
)

# ==================================================
# DASHBOARD
# ==================================================

if menu == "Dashboard":

    st.title("📰 Fake News Detection Dashboard")

    if not df.empty:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Articles",
            len(df)
        )

        col2.metric(
            "Fake Articles",
            len(df[df["label"] == 1])
        )

        col3.metric(
            "Real Articles",
            len(df[df["label"] == 0])
        )

        col4.metric(
            "Subjects",
            df["subject"].nunique()
        )

        st.subheader("Subject Distribution")

        st.bar_chart(
            df["subject"].value_counts()
        )

    else:
        st.warning("Dataset not loaded.")

# ==================================================
# FAKE NEWS DETECTION
# ==================================================

elif menu == "Fake News Detection":

    st.title("🔍 Fake News Detector")

    input_mode = st.radio(
        "Choose Input Type",
        [
            "Paste Text",
            "News URL",
            "Upload File"
        ]
    )

    # ------------------------
    # TEXT
    # ------------------------

    if input_mode == "Paste Text":

        article = st.text_area(
            "Paste News Article",
            height=300
        )

        if st.button("Analyze Text"):

            if article.strip():

                label, confidence = predict_article(article)

                display_prediction(
                    label,
                    confidence
                )

            else:
                st.warning("Please enter some text.")

    # ------------------------
    # URL
    # ------------------------

    elif input_mode == "News URL":

        url = st.text_input(
            "Enter News URL"
        )

        if st.button("Extract & Analyze"):

            if url.strip():

                data = extract_news(url)

                if data:

                    st.subheader(data["title"])

                    st.write(
                        "Authors:",
                        data["authors"]
                    )

                    st.write(
                        "Published:",
                        data["date"]
                    )

                    st.text_area(
                        "Extracted Article",
                        data["text"][:5000],
                        height=300
                    )

                    label, confidence = predict_article(
                        data["text"]
                    )

                    display_prediction(
                        label,
                        confidence
                    )

                    domain = urlparse(
                        url
                    ).netloc.replace(
                        "www.",
                        ""
                    )

                    score = SOURCE_SCORES.get(
                        domain,
                        50
                    )

                    st.metric(
                        "Source Credibility",
                        f"{score}/100"
                    )

            else:
                st.warning("Please enter a valid URL.")

    # ------------------------
    # FILE UPLOAD
    # ------------------------

    elif input_mode == "Upload File":

        uploaded_file = st.file_uploader(
            "Upload TXT or PDF",
            type=[
                "txt",
                "pdf"
            ]
        )

        if uploaded_file:

            text = ""

            if uploaded_file.name.endswith(".txt"):

                text = uploaded_file.read().decode(
                    "utf-8"
                )

            elif uploaded_file.name.endswith(".pdf"):

                with pdfplumber.open(uploaded_file) as pdf:

                    for page in pdf.pages:

                        page_text = page.extract_text()

                        if page_text:
                            text += page_text

            st.text_area(
                "Extracted Content",
                text[:5000],
                height=300
            )

            if st.button("Analyze File"):

                if text.strip():

                    label, confidence = predict_article(
                        text
                    )

                    display_prediction(
                        label,
                        confidence
                    )

                else:
                    st.warning("No readable text found in the uploaded file.")
# ==================================================
# LABELING
# ==================================================

elif menu == "News Labeling":

    st.title("🏷️ News Labeling")

    if not df.empty:

        sample = df.sample(1).iloc[0]

        st.subheader(sample["title"])

        st.write(sample["text"][:3000])

        label = st.radio(
            "Select Label",
            [
                "Real",
                "Fake",
                "Misleading",
                "Satire",
                "Biased"
            ]
        )

        comment = st.text_area(
            "Reviewer Notes"
        )

        if st.button("Save Annotation"):

            row = pd.DataFrame({

                "title": [sample["title"]],
                "label": [label],
                "comment": [comment]

            })

            file_exists = os.path.exists("annotations.csv")

            row.to_csv(
                "annotations.csv",
                mode="a",
                header=not file_exists,
                index=False
            )

            st.success("✅ Annotation Saved Successfully")

    else:

        st.warning("Dataset not loaded.")

# ==================================================
# ANALYTICS
# ==================================================

elif menu == "Analytics":

    st.title("📊 Analytics")

    if not df.empty:

        counts = df["label"].value_counts()

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.pie(
            counts,
            labels=["Real", "Fake"],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.axis("equal")

        st.pyplot(fig)

        st.subheader("☁️ Word Cloud")

        sample_text = " ".join(
            df["text"].sample(
                min(2000, len(df))
            )
        )

        wc = WordCloud(
            width=1000,
            height=500,
            background_color="white"
        ).generate(sample_text)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.imshow(wc)

        ax.axis("off")

        st.pyplot(fig)

    else:

        st.warning("Dataset not loaded.")

# ==================================================
# DATASET EXPLORER
# ==================================================

elif menu == "Dataset Explorer":

    st.title("📚 Dataset Explorer")

    if not df.empty:

        st.dataframe(df.head(100))

        st.write("Shape:", df.shape)

        st.write("Columns:")

        st.write(df.columns.tolist())

        st.download_button(
            label="📥 Download Dataset",
            data=df.to_csv(index=False),
            file_name="news_dataset.csv",
            mime="text/csv"
        )

    else:

        st.warning("Dataset not loaded.")

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;color:gray'>
        📰 <b>NewsGuard AI</b><br>
        Fake News Detection using Deep Learning (BiGRU + Attention)<br>
        Developed using Streamlit & TensorFlow
    </div>
    """,
    unsafe_allow_html=True
)



# In[ ]:




