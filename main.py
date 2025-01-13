import streamlit as st
import pickle
import pandas as pd

# Charger le modèle et les pipelines
@st.cache_resource
def load_model_and_pipeline():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('pipeline.pkl', 'rb') as pipeline_file:
        pipeline = pickle.load(pipeline_file)
    return model, pipeline

# Charger le modèle et le pipeline
model, pipeline = load_model_and_pipeline()

# Interface utilisateur
st.title("Application de Prédiction Interactive")
st.write("Saisissez vos données pour générer une prédiction.")

# Exemple de formulaire pour la saisie utilisateur
age = st.number_input("Âge", min_value=0, max_value=120, value=30, step=1)
genre = st.selectbox("Genre", options=["Homme", "Femme"])
salaire_annuel = st.number_input("Salaire Annuel (€)", min_value=0, step=1000, value=50000)
score_credit = st.slider("Score de Crédit", min_value=0, max_value=1000, value=500)

# Conversion des données utilisateur en DataFrame
user_data = {
    "age": [age],
    "genre": [genre],
    "salaire_annuel": [salaire_annuel],
    "score_credit": [score_credit],
}
input_df = pd.DataFrame(user_data)

# Prétraitement des données
st.write("Données brutes saisies :")
st.write(input_df)

try:
    preprocessed_data = pipeline.transform(input_df)
except Exception as e:
    st.error(f"Erreur lors du prétraitement des données : {e}")
    preprocessed_data = None

# Génération de la prédiction
if preprocessed_data is not None:
    prediction = model.predict(preprocessed_data)
    st.write("**Prédiction générée :**")
    st.write(prediction)
else:
    st.warning("Impossible de générer une prédiction en raison d'une erreur dans le prétraitement.")

# Ajouter une explication si possible
if st.checkbox("Afficher les probabilités ou l'explication (si disponible)"):
    try:
        prediction_proba = model.predict_proba(preprocessed_data)
        st.write("Probabilités des classes :")
        st.write(prediction_proba)
    except AttributeError:
        st.warning("Le modèle ne fournit pas de probabilités ou d'explications.")
