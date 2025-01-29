import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Chargement du modèle sauvegardé
with open("Lasso_pipeline.pkl", "rb") as file:
    lasso_model = pickle.load(file)

# Titre de l'application
st.title("Devis des frais d'assurance Assur'Aimant")
st.image('DALL·E 2025-01-17 09.18.06 - A professional and modern illustration representing insurance and data predictions. The image includes elements such as a family under an umbrella (sy.webp')

# Description
st.write("""
Cette application prédit les frais d'assurance en fonction des informations saisies.  
Veuillez remplir les informations ci-dessous.""")

# Entrées utilisateur
age = st.number_input("Âge", min_value=18, max_value=100, value=30, step=1)
sexe = st.selectbox("Sexe", ["male", "female"])
poids = st.number_input("Poids (en kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
taille = st.number_input("Taille (en cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)

# Calcul automatique de l'IMC
imc = poids / ((taille / 100) ** 2)
st.write(f"Votre IMC calculé est : {imc:.2f}")

enfants = st.number_input("Enfants", min_value=0, max_value=10, value=0, step=1)
fumeur = st.selectbox("Fumeur", ["yes", "no"])
region = st.selectbox("Région", ["northwest", "northeast", "southwest", "southeast"])

# Transformation des données en DataFrame
donnees_utilisateur = pd.DataFrame({
    "Âge": [age],
    "Sexe": [sexe],
    "IMC": [imc],
    "Enfants": [enfants],
    "Fumeur": [fumeur],
    "Région": [region]
})

# Prédiction
if st.button("Prédire les frais d'assurance"):
    prediction = lasso_model.predict(donnees_utilisateur)
    st.success(f"Les frais d'assurance estimés sont de {prediction[0]:.2f} $")

# Footer
st.write("---")
st.write("Merci de nous contacter pour plus d'informations au 3630 🎅.")
