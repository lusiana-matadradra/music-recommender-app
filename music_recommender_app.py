import streamlit as st
import joblib
import pandas as pd

# Load models and encoders
model_genre = joblib.load("mbti_genre_model.pkl")
model_artist = joblib.load("artist_model.pkl")
le_genre = joblib.load("le_genre.pkl")
le_artist = joblib.load("le_artist.pkl")

# Streamlit UI
st.title("🎧 Music Genre Recommender")
st.write("Answer a few personality questions and we’ll recommend a music genre and some artists just for you!")

st.subheader("🧠 Tell us about yourself")

# MBTI Inputs
mbti_options = ["", "Extraversion", "Introversion"]
info_options = ["Sensing", "Intuition"]
decision_options = ["Thinking", "Feeling"]
lifestyle_options = ["Judging", "Perceiving"]
tempo_options = ["Slow/Calm", "Medium", "Fast/Energetic"]

social = st.selectbox("When it comes to socialising:", mbti_options, index=0, help="Choose whether you prefer alone time or big groups")
if social:
    st.caption("I prefer spending time alone or with a small group of close friends (Introversion)" if social == "Introversion" else "I gain energy from being around others (Extraversion)")

info = st.radio("When processing information:", info_options, help="How do you trust and process information?")
if info:
    st.caption("I trust facts, data, and real experiences (Sensing)" if info == "Sensing" else "I rely on intuition and big-picture thinking (Intuition)")

decision = st.radio("When making decisions:", decision_options, help="Do you lean more on logic or feelings?")
if decision:
    st.caption("I prioritise logic and objectivity (Thinking)" if decision == "Thinking" else "I consider emotions and values (Feeling)")

lifestyle = st.radio("When planning my day or tasks:", lifestyle_options, help="Do you prefer structure or flexibility?")
if lifestyle:
    st.caption("I like structure, planning, and sticking to schedules (Judging)" if lifestyle == "Judging" else "I prefer to stay open to new options (Perceiving)")

tempo = st.radio("Preferred music tempo:", tempo_options)

# Check if MBTI is fully selected
if all([social, info, decision, lifestyle]):
    mbti_input = (social[0] + info[0] + decision[0] + lifestyle[0]).upper()
    st.write(f"**Your MBTI type:** {mbti_input}")

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
else:
    st.warning("Please complete all personality questions before continuing.")
