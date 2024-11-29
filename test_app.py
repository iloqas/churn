import pytest
from app import app  # Assurez-vous que vous importez votre application Flask correctement

# Créez un client de test à utiliser pour les tests
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Test avec des données valides
def test_valid_data(client):
    data = {
        'Age': 30,
        'Total_Purchase': 48,
        'Years': 5,
        'Num_Sites': 2,
        'Account_Manager': 1
    }

    response = client.post('/predict', data=data)
    
    # Vérifier que le code de statut est 200
    assert response.status_code == 200
    


# Test avec des données manquantes
def test_missing_field(client):
    data = {
        'Age': 30,
        'Total_Purchase': 48,
        'Num_Sites': 2,
        'Account_Manager': 1
    }  # Le champ 'Years' est manquant

    response = client.post('/predict', data=data)
    
    # Vérifier que le code de statut est 400 (erreur)
    assert response.status_code == 400
    
    # Vérifier que l'erreur contient un message d'erreur
    response_json = response.get_json()
    assert 'error' in response_json

# Test avec des types de données incorrects
def test_invalid_data_type(client):
    data = {
        'Age': "thirty",  # Age n'est pas un nombre
        'Total_Purchase': 48,
        'Years': 5,
        'Num_Sites': 2,
        'Account_Manager': 1
    }

    response = client.post('/predict', data=data)
    
    # Vérifier que le code de statut est 400 (erreur)
    assert response.status_code == 400
    
    # Vérifier que l'erreur contient un message d'erreur
    response_json = response.get_json()
    assert 'error' in response_json

# Test avec des données vides
def test_empty_data(client):
    data = {}

    response = client.post('/predict', data=data)
    
    # Vérifier que le code de statut est 400 (erreur)
    assert response.status_code == 400
    
    # Vérifier que l'erreur contient un message d'erreur
    response_json = response.get_json()
    assert 'error' in response_json
