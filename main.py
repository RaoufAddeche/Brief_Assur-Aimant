import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Chargement du modèle exporté
@st.cache_resource
def charger_modele():
    with open("modele.pkl", "rb") as file:
        modele = pickle.load(file)
    return modele

modele = charger_modele()

# Titre de l'application
st.title("Prédiction des Primes d'Assurance")

st.write("Cette application permet de prédire les primes d'assurance en fonction des données démographiques de l'utilisateur.")

# Saisie des informations utilisateur
st.sidebar.header("Entrez vos informations :")

age = st.sidebar.number_input("Âge", min_value=18, max_value=100, value=30)
sexe = st.sidebar.selectbox("Sexe", options=["Homme", "Femme"])
bmi = st.sidebar.number_input("IMC (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
enfants = st.sidebar.number_input("Nombre d'enfants", min_value=0, max_value=10, value=0, step=1)
fumeur = st.sidebar.selectbox("Fumeur", options=["Oui", "Non"])
region = st.sidebar.selectbox("Région", options=["Sud-Ouest", "Sud-Est", "Nord-Ouest", "Nord-Est"])

# Préparation des données
st.write("### Données saisies :")
donnees_utilisateur = {
    "Âge": age,
    "Sexe": sexe,
    "IMC": bmi,
    "Nombre d'enfants": enfants,
    "Fumeur": fumeur,
    "Région": region,
}

donnees_df = pd.DataFrame([donnees_utilisateur])
st.write(donnees_df)

# Encodage et préparation des données (doit correspondre au pipeline utilisé dans le modèle)
@st.cache_data
def preprocess_data(donnees):
    donnees = donnees.copy()
    donnees["Sexe"] = donnees["Sexe"].map({"Homme": 1, "Femme": 0})
    donnees["Fumeur"] = donnees["Fumeur"].map({"Oui": 1, "Non": 0})
    donnees = pd.get_dummies(donnees, columns=["Région"], drop_first=True)
    return donnees

donnees_pretraitees = preprocess_data(donnees_df)

# Standardisation (si nécessaire)
scaler = StandardScaler()
donnees_finales = scaler.fit_transform(donnees_pretraitees)

# Prédiction
if st.button("Prédire"):
    prediction = modele.predict(donnees_finales)
    st.write("### Résultat de la Prédiction :")
    st.success(f"Votre prime d'assurance estimée est : {prediction[0]:.2f} €")

