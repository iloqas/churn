from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

# Configurer le logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Charger le modèle de régression logistique sauvegardé
try:
    model = joblib.load('data/model.pkl')
    logging.info("Modèle chargé avec succès.")
except FileNotFoundError:
    logging.error("Le fichier 'model.pkl' est introuvable. Assurez-vous qu'il est placé dans le dossier 'data'.")
    exit(1)
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle : {str(e)}")
    exit(1)

# Initialiser l'application Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire HTML
        age = float(request.form.get('Age', 0))
        total_purchase = float(request.form.get('Total_Purchase', 0))
        years = float(request.form.get('Years', 0))
        num_sites = int(request.form.get('Num_Sites', 0))
        account_manager = int(request.form.get('Account_Manager', 0))  # Si ce champ n'existe pas, supprimez-le

        # Créer un tableau numpy pour les données de prédiction
        features = np.array([[age, total_purchase, years, num_sites, account_manager]])

        # Effectuer la prédiction
        prediction = model.predict(features)
        result = "Churn" if prediction[0] == 1 else "Non Churn"

        logging.info(f"Prédiction effectuée avec succès : {result}")

        # Retourner une réponse JSON avec le résultat de la prédiction
        return jsonify({'prediction': result})

    except ValueError as ve:
        logging.error(f"Erreur de validation des données : {ve}")
        return jsonify({'error': "Veuillez entrer des valeurs valides."}), 400
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({'error': "Une erreur est survenue. Veuillez réessayer."}), 500

# Fonction pour lancer le serveur Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5012, debug=True)  # Passer debug=False en production
