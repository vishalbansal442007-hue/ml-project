# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Movie Recommender System", layout="wide")

st.title("🎬 Movie Recommender System using K-Means Clustering")

# =========================
# LOAD DATASET
# =========================

df1 = pd.read_csv("ML_PROJECT_1_updated.csv")

df1 = df1.drop(['Timestamp', 'Email Address'], axis=1)

# =========================
# ENCODING
# =========================

# Gender
gender = df1["What's your gender ?"].unique()
OHE_gender = OneHotEncoder(categories=[gender], drop='first', sparse_output=False)
X1_encoded_OHE = OHE_gender.fit_transform(df1[["What's your gender ?"]])

# Selection criteria
selection_criteria = df1["First thing you look while selecting a movie to watch."].unique()
OHE_selection = OneHotEncoder(categories=[selection_criteria], drop='first', sparse_output=False)
X2_encoded_OHE = OHE_selection.fit_transform(
    df1[["First thing you look while selecting a movie to watch."]]
)

# Language
language_criteria = df1['Which language do you prefer for watching movie?  '].unique()
OHE_language = OneHotEncoder(categories=[language_criteria], drop='first', sparse_output=False)
X3_encoded_OHE = OHE_language.fit_transform(
    df1[['Which language do you prefer for watching movie?  ']]
)

# Industry
industry_criteria = df1['Which movie industry do you prefer ?'].unique()
OHE_industry = OneHotEncoder(categories=[industry_criteria], drop='first', sparse_output=False)
X4_encoded_OHE = OHE_industry.fit_transform(
    df1[['Which movie industry do you prefer ?']]
)

# Frequency
frequency_criteria = df1['How many movies do you watch per month?'].unique()
OHE_frequency = OneHotEncoder(categories=[frequency_criteria], drop='first', sparse_output=False)
X5_encoded_OHE = OHE_frequency.fit_transform(
    df1[['How many movies do you watch per month?']]
)

# Duration
duration_criteria = df1['What movie duration do you prefer?'].unique()
OHE_duration = OneHotEncoder(categories=[duration_criteria], drop='first', sparse_output=False)
X6_encoded_OHE = OHE_duration.fit_transform(
    df1[['What movie duration do you prefer?']]
)

# Ordinal Encoding
OE = OrdinalEncoder(categories=[['Under 18', '16 to 18', 'Above 18']])

X_encoded_OE = OE.fit_transform(df1.iloc[:, [0]].values)

# Ratings
Array_ratings = df1.iloc[:, 7:15].values

# Final Data
X_final = np.hstack((
    X_encoded_OE,
    X1_encoded_OHE,
    X2_encoded_OHE,
    X3_encoded_OHE,
    X4_encoded_OHE,
    X5_encoded_OHE,
    X6_encoded_OHE,
    Array_ratings
))

# =========================
# SCALING + PCA + KMEANS
# =========================

SS = StandardScaler()
X_scaled = SS.fit_transform(X_final)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_pca)

# =========================
# USER INPUT
# =========================

st.sidebar.header("Enter Your Preferences")

age = st.sidebar.selectbox(
    "Age",
    ['Under 18', '16 to 18', 'Above 18']
)

gender_input = st.sidebar.selectbox(
    "Gender",
    df1["What's your gender ?"].unique()
)

selection_input = st.sidebar.selectbox(
    "What do you look for in movies?",
    df1["First thing you look while selecting a movie to watch."].unique()
)

language_input = st.sidebar.selectbox(
    "Preferred Language",
    df1['Which language do you prefer for watching movie?  '].unique()
)

industry_input = st.sidebar.selectbox(
    "Preferred Industry",
    df1['Which movie industry do you prefer ?'].unique()
)

frequency_input = st.sidebar.selectbox(
    "Movies watched per month",
    df1['How many movies do you watch per month?'].unique()
)

duration_input = st.sidebar.selectbox(
    "Preferred Movie Duration",
    df1['What movie duration do you prefer?'].unique()
)

st.sidebar.subheader("Rate Movies (1-5)")

r1 = st.sidebar.slider("Romantic", 1, 5, 3)
r2 = st.sidebar.slider("Comedy", 1, 5, 3)
r3 = st.sidebar.slider("Horror", 1, 5, 3)
r4 = st.sidebar.slider("Thriller", 1, 5, 3)
r5 = st.sidebar.slider("Biopic", 1, 5, 3)
r6 = st.sidebar.slider("Science-fiction", 1, 5, 3)
r7 = st.sidebar.slider("Action-Drama", 1, 5, 3)
r8 = st.sidebar.slider("Inspirational", 1, 5, 3)

# =========================
# PREDICT BUTTON
# =========================

if st.button("Recommend Movies"):

    # Ordinal
    age_encoded = OE.transform([[age]])

    # OHE
    gender_encoded = OHE_gender.transform([[gender_input]])
    selection_encoded = OHE_selection.transform([[selection_input]])
    language_encoded = OHE_language.transform([[language_input]])
    industry_encoded = OHE_industry.transform([[industry_input]])
    frequency_encoded = OHE_frequency.transform([[frequency_input]])
    duration_encoded = OHE_duration.transform([[duration_input]])

    ratings = np.array([[r1, r2, r3, r4, r5, r6, r7, r8]])

    final_input = np.hstack((
        age_encoded,
        gender_encoded,
        selection_encoded,
        language_encoded,
        industry_encoded,
        frequency_encoded,
        duration_encoded,
        ratings
    ))

    # Scale + PCA
    scaled_input = SS.transform(final_input)
    pca_input = pca.transform(scaled_input)

    # Predict cluster
    cluster = kmeans.predict(pca_input)[0]

    st.success(f"Predicted Cluster: {cluster}")

    # =========================
    # MOVIE RECOMMENDATIONS
    # =========================

    if cluster == 0:
        st.header("🎥 Highly Engaged Bollywood-Hindi Frequent Viewers")
        st.write([
            "Kabir Singh",
            "Tanhaji",
            "Bajrangi Bhaijaan",
            "Sultan",
            "War",
            "KGF",
            "Pushpa",
            "RRR"
        ])

    elif cluster == 1:
        st.header("🎥 Balanced Multi-Genre Viewers")
        st.write([
            "3 Idiots",
            "Drishyam",
            "Andhadhun",
            "Interstellar",
            "The Dark Knight",
            "Avengers"
        ])

    elif cluster == 2:
        st.header("🎥 Critical Diverse Viewers")
        st.write([
            "Shutter Island",
            "Inception",
            "Golmaal",
            "Stree",
            "Bahubali",
            "Vikram"
        ])

    elif cluster == 3:
        st.header("🎥 Selective Hindi Viewers")
        st.write([
            "Mary Kom",
            "Dangal",
            "Padman",
            "Taare Zameen Par",
            "English Vinglish",
            "Queen"
        ])

    elif cluster == 4:
        st.header("🎥 South Indian & Cast-Focused Viewers")
        st.write([
            "RRR",
            "Bahubali",
            "Vikram",
            "Kantara",
            "KGF",
            "Drishyam 2",
            "Salaar"
        ])
