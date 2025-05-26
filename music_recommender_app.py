import streamlit as st
import joblib
import pandas as pd

# Load models and encoders
model_genre = joblib.load("mbti_genre_model.pkl")
model_artist = joblib.load("artist_model.pkl")
le_genre = joblib.load("le_genre.pkl")
le_artist = joblib.load("le_artist.pkl")

# Streamlit UI
st.title("🎧 Music Genre & Artist Recommender")
st.write("Answer a few MBTI-style questions and we'll suggest a music genre and an artist!")

# MBTI Inputs
col1, col2 = st.columns(2)

with col1:
    social = st.radio("When it comes to socialising:", ["Extraversion", "Introversion"])
    info = st.radio("When processing information:", ["Sensing", "Intuition"])

with col2:
    decision = st.radio("When making decisions:", ["Thinking", "Feeling"])
    lifestyle = st.radio("Your approach to life:", ["Judging", "Perceiving"])

# Form MBTI string
mbti_input = (social[0] + info[0] + decision[0] + lifestyle[0]).upper()
st.write(f"**Your MBTI type:** {mbti_input}")

# Predict
if st.button("Get Recommendation"):
    try:
        mbti_df = pd.DataFrame([[mbti_input]], columns=['MBTI'])

        # Predict genre and artist
        genre_encoded = model_genre.predict(mbti_df)[0]
        artist_encoded = model_artist.predict(mbti_df)[0]

        # Decode labels
        predicted_genre = le_genre.inverse_transform([genre_encoded])[0]
        recommended_artist = le_artist.inverse_transform([artist_encoded])[0]

        # Display results
        st.success(f"🎶 Recommended Genre: **{predicted_genre}**")
        st.info(f"🎤 Suggested Artist Group: **{recommended_artist}**")

    except Exception as e:
        st.error(f"Oops! Something went wrong: {e}")

st.caption("Made with ❤️ for Project 3 - 297.201 Data Science")
