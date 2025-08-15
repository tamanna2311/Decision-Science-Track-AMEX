"""
File: test_api.py
Purpose: Automatically verify that the API handles requests correctly.
Why it exists: If we change code in the future, we might accidentally break the API. 
               This test runs instantly and proves the endpoint functions end-to-end.
What it imports: pytest, FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

# Create a fake client to send HTTP requests to our app without starting a real server
client = TestClient(app)

def test_rank_offers_success():
    """
    Tests if a valid payload receives a valid 200 OK sorted ranking response.
    """
    # A fake valid payload matching our Schemas
    payload = {
        "user": {
            "user_id": 105,
            "device_type": "mobile_app",
            "user_segment": "high_spender",
            "previous_clicks": 5,
            "previous_impressions": 50,
            "time_since_last_activity": 2.5,
            "transaction_count": 12
        },
        "candidate_offers": [
            {"offer_id": 201, "offer_category": "dining"},
            {"offer_id": 202, "offer_category": "travel"},
            {"offer_id": 203, "offer_category": "retail"}
        ]
    }
    
    # Send a POST request directly to the endpoint
    response = client.post("/rank_offers", json=payload)
    
    # Assertions block (checking if our expectations are met)
    assert response.status_code == 200, "API did not return a successful 200 status code."
    
    data = response.json()
    assert "ranked_offers" in data, "Response is missing 'ranked_offers' key."
    
    ranked_offers = data["ranked_offers"]
    assert len(ranked_offers) == 3, "API dropped some candidate offers."
    
    # Check that they are sorted by rank
    assert ranked_offers[0]["rank"] == 1
    assert ranked_offers[1]["rank"] == 2
    
    # Check that scores are monotonically decreasing (Rank 1 score >= Rank 2 score)
    assert ranked_offers[0]["score"] >= ranked_offers[1]["score"]
