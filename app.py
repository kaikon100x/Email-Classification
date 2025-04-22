import os
import uvicorn
import argparse
from api import app
from models import EmailClassifier, prepare_dummy_data


def train_model():
    """Train and save the email classification model"""
    texts, labels = prepare_dummy_data()

    print("Training email classification model...")
    classifier = EmailClassifier(model_type="traditional")
    classifier.train(texts, labels)
    classifier.save_model("email_classifier.pkl")
    print("Model trained and saved successfully!")


def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description="Email Classification API")
    parser.add_argument("--train", action="store_true", help="Train the model before starting the API")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API on")

    args = parser.parse_args()

    # Train model if requested or if model doesn't exist
    if args.train or not os.path.exists("email_classifier.pkl"):
        train_model()

    # Start the API server
    print(f"Starting API server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()