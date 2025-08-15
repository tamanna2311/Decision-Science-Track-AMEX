"""
File: schemas.py
Purpose: Define the exact shape and type of data the API expects.
Why it exists: If a frontend sends 'age: twenty' instead of 'age: 20', it breaks the code. 
               Pydantic schemas automatically catch these errors before they reach our ML model.
What it imports: pydantic.
Which files use it: api/main.py.
Inputs: Raw JSON from the internet.
Outputs: Validated Python classes.
"""

from pydantic import BaseModel, conlist
from typing import List

class UserContext(BaseModel):
    """
    Data about the user asking for recommendations.
    """
    user_id: int
    device_type: str
    user_segment: str
    previous_clicks: int = 0
    previous_impressions: int = 0
    time_since_last_activity: float = 0.0
    transaction_count: int = 0

class OfferContext(BaseModel):
    """
    Data about a single offer that we are considering showing the user.
    """
    offer_id: int
    offer_category: str

class RankRequest(BaseModel):
    """
    The full request payload. One user, multiple candidate offers to rank.
    """
    user: UserContext
    # We enforce that there is at least 1 candidate offer provided
    candidate_offers: conlist(OfferContext, min_length=1)

class RankedOfferResponse(BaseModel):
    """
    Data going out. The exact format the frontend expects back.
    """
    offer_id: int
    score: float
    rank: int

class RankResponse(BaseModel):
    ranked_offers: List[RankedOfferResponse]
