from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os

# Import our utility functions and models
from utils import PIIMasker, preprocess_text
from models import EmailClassifier, prepare_dummy_data


# Define request/response models
class EmailRequest(BaseModel):
    email_body: str


class MaskedEntity(BaseModel):
    position: List[int]
    classification: str
    entity: str


class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[MaskedEntity]
    masked_email: str
    category_of_the_email: str


# Initialize FastAPI app
app = FastAPI(title="Email Classification API",
              description="API for classifying emails and masking PII",
              version="1.0.0")

# Initialize PII masker
pii_masker = PIIMasker()

# Load or train email classifier
model_path = "email_classifier.pkl"
if os.path.exists(model_path):
    email_classifier = EmailClassifier.load_model(model_path)
else:
    # Train with dummy data
    texts, labels = prepare_dummy_data()
    email_classifier = EmailClassifier(model_type="traditional")
    email_classifier.train(texts, labels)
    email_classifier.save_model(model_path)


@app.post("/classify-email", response_model=EmailResponse)
async def classify_email(email_request: EmailRequest = Body(...)):
    """
    Classify an email and mask PII

    Args:
        email_request: Email body to process

    Returns:
        dict: Classification results and masked email
    """
    try:
        # Get email text
        email_text = email_request.email_body

        # Mask PII
        masked_email, entities = pii_masker.mask_pii(email_text)

        # Preprocess text for classification
        processed_text = preprocess_text(masked_email)

        # Classify email
        category = email_classifier.predict(processed_text)

        # Format response
        response = {
            "input_email_body": email_text,
            "list_of_masked_entities": entities,
            "masked_email": masked_email,
            "category_of_the_email": category
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing email: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Email Classification API is running",
        "endpoints": {
            "/classify-email": "POST endpoint to classify emails and mask PII"
        }
    }