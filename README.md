# Raouf Addeche - Brief Assur-Aimant

Ce projet vise à analyser, modéliser et prédire des données liées à l'assurance en utilisant des techniques de machine learning. Il inclut des étapes de nettoyage des données, de modélisation, et d'évaluation des performances des modèles.

## Structure du projet

Voici la structure des fichiers et dossiers du projet, avec une description de leur rôle :

```
raoufaddeche-brief_assur-aimant/
├── README.md                     # Documentation du projet
├── analyse.ipynb                 # Analyse exploratoire des données
├── course+-+Quiz+P2-V2.csv       # Données brutes (quiz ou autres informations)
├── DALL·E 2025-01-17 09.18.06...# Illustration représentant l'assurance et les prédictions
├── data_nettoyer.csv             # Données nettoyées prêtes pour la modélisation
├── data_set.csv                  # Jeu de données brut
├── Lasso_pipeline.pkl            # Modèle sauvegardé (pipeline Lasso)
├── linear_regression_model.pkl   # Modèle de régression linéaire sauvegardé
├── main.py                       # Script principal pour la modélisation et les prédictions
├── modelisation.ipynb            # Notebook pour la création et l'évaluation des modèles
├── nettoyage.ipynb               # Notebook pour le nettoyage des données
├── note.txt                      # Notes diverses sur le projet
├── personnes(1).csv              # Données supplémentaires (informations sur les personnes)
├── requirements.txt              # Liste des dépendances Python nécessaires
├── test.csv                      # Jeu de données de test
└── training.ipynb                # Notebook pour l'entraînement des modèles
```

## Description des fichiers principaux

- **`analyse.ipynb`** : Contient l'analyse exploratoire des données (EDA), avec des visualisations et des statistiques descriptives.
- **`nettoyage.ipynb`** : Étapes de nettoyage des données, comme le traitement des valeurs manquantes et la normalisation.
- **`modelisation.ipynb`** : Implémentation des modèles de machine learning, y compris la régression linéaire et d'autres algorithmes.
- **`main.py`** : Script principal pour exécuter les étapes de modélisation et de prédiction.
- **`data_nettoyer.csv`** : Jeu de données nettoyé, prêt pour l'entraînement des modèles.
- **`Lasso_pipeline.pkl`** et **`linear_regression_model.pkl`** : Modèles sauvegardés pour une utilisation ultérieure.
- **`requirements.txt`** : Liste des bibliothèques Python nécessaires pour exécuter le projet.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les dépendances nécessaires. Vous pouvez les installer avec la commande suivante :

```bash
pip install -r requirements.txt
```

## Utilisation

1. **Nettoyage des données** :
   - Ouvrez et exécutez le notebook `nettoyage.ipynb` pour préparer les données.

2. **Analyse exploratoire** :
   - Utilisez le notebook `analyse.ipynb` pour explorer les données et comprendre leurs caractéristiques.

3. **Modélisation** :
   - Exécutez le notebook `modelisation.ipynb` ou le script `main.py` pour entraîner et évaluer les modèles.

4. **Prédictions** :
   - Chargez un modèle sauvegardé (par exemple, `linear_regression_model.pkl`) pour effectuer des prédictions sur de nouvelles données.

## Résultats

Les résultats des modèles, y compris les métriques d'évaluation (MSE, R², etc.), sont disponibles dans les notebooks ou affichés dans la console lors de l'exécution de `main.py`.

## Auteur

Raouf Addeche

---

Si vous avez des questions ou des suggestions, n'hésitez pas à me contacter !
