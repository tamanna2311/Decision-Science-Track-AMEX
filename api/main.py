"""
File: main.py
Purpose: The entry point for the FastAPI server.
Why it exists: We need an HTTP server so that other applications (like a web frontend or mobile app) 
               can send data over the internet and get a ranked list of offers back. 
               A model on a disk is useless if nobody can talk to it.
What it imports: FastAPI, our schemas, and our inference engine (predict.py).
Which files use it: Uvicorn runs this file.
Role in the pipeline: The external face of the ML system.
"""

from fastapi import FastAPI, HTTPException
from api.schemas import RankRequest, RankResponse
from src.predict import rank_candidate_offers, load_artifacts
from src.utils import get_logger

logger = get_logger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="Amex Digital Offer Personalization API",
    description="Ranks a list of candidate offers for a specific user using an XGBoost model.",
    version="1.0.0"
)

# Startup event: We want to load the ML model into memory exactly once when the server boots.
# If we loaded it inside the endpoint, every single request would take seconds instead of milliseconds.
@app.on_event("startup")
async def startup_event():
    logger.info("Server starting up. Loading ML models into memory...")
    try:
        load_artifacts()
        logger.info("Models loaded successfully. Ready to serve requests.")
    except Exception as e:
        logger.error(f"Failed to load models upon startup: {e}")
        # Note: If artifacts are missing, the server will start but endpoints will fail.
        # Ensure `python run_pipeline.py` has been run first.

@app.post("/rank_offers", response_model=RankResponse)
async def rank_offers_endpoint(request: RankRequest):
    """
    The core API endpoint.
    Expects a JSON body with 'user' info and a list of 'candidate_offers'.
    Returns a sorted list of offers with their click probability score.
    """
    logger.info(f"Received ranking request for user_id: {request.user.user_id}")
    
    try:
        # Pydantic automatically converted the raw JSON into nicer Python objects.
        # We turn them into dicts because our `rank_candidate_offers` function expects dicts.
        user_dict = request.user.dict()
        offers_list = [offer.dict() for offer in request.candidate_offers]
        
        # 1. Run the ML inference
        ranked_results = rank_candidate_offers(user_dict, offers_list)
        
        # 2. Return the response
        return RankResponse(ranked_offers=ranked_results)
        
    except Exception as e:
        logger.error(f"Error during ranking inference: {str(e)}")
        # Return a 500 Internal Server Error so the frontend knows we failed gracefully
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")

# To run locally: uvicorn api.main:app --reload
