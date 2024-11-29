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
    

