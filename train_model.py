#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    GRU,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# ===================================
# CREATE MODELS DIRECTORY
# ===================================

os.makedirs("models", exist_ok=True)

# ===================================
# LOAD DATA
# ===================================

print("Loading dataset...")

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 1
true["label"] = 0

df = pd.concat([fake, true], ignore_index=True)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ===================================
# CLEAN TEXT
# ===================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

df["content"] = (
    df["title"] + " " + df["text"]
)

df["content"] = df["content"].apply(clean_text)

# ===================================
# TOKENIZATION
# ===================================

MAX_WORDS = 50000
MAX_LEN = 500

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(df["content"])

X = tokenizer.texts_to_sequences(
    df["content"]
)

X = pad_sequences(
    X,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

y = df["label"].values

# ===================================
# TRAIN TEST SPLIT
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===================================
# MODEL
# ===================================

inputs = Input(shape=(MAX_LEN,))

embedding = Embedding(
    input_dim=MAX_WORDS,
    output_dim=128
)(inputs)

gru1 = Bidirectional(
    GRU(
        128,
        return_sequences=True,
        dropout=0.2
    )
)(embedding)

gru2 = Bidirectional(
    GRU(
        64,
        return_sequences=True,
        dropout=0.2
    )
)(gru1)

attention = Attention()(
    [gru2, gru2]
)

pool = GlobalAveragePooling1D()(
    attention
)

dense = Dense(
    64,
    activation="relu"
)(pool)

drop = Dropout(0.3)(
    dense
)

outputs = Dense(
    1,
    activation="sigmoid"
)(drop)

model = Model(
    inputs=inputs,
    outputs=outputs
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===================================
# CALLBACKS
# ===================================

checkpoint = ModelCheckpoint(
  "models/bigru_attention_model.keras",
  monitor="val_accuracy",
  mode="max",
  save_best_only=True,
  verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# ===================================
# TRAIN
# ===================================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=2,
    batch_size=64,
    callbacks=[
        checkpoint,
        early_stop
    ]
)

# ===================================
# EVALUATE
# ===================================

preds = (
    model.predict(X_test) > 0.5
).astype(int)

print(
    classification_report(
        y_test,
        preds
    )
)

# ===================================
# SAVE FINAL MODEL
# ===================================

model.save(
    "models/final_bigru_attention.keras"
)

# ===================================
# SAVE TOKENIZER
# ===================================

with open(
    "models/tokenizer.pkl",
    "wb"
) as f:

    pickle.dump(
        tokenizer,
        f
    )

print("\nTraining Complete")
print("Saved:")
print("models/bigru_attention_model.keras")
print("models/final_bigru_attention.keras")
print("models/tokenizer.pkl")


# In[ ]:




