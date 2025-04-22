import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import os
from typing import List, Dict, Any, Union, Tuple
import json

# For Hugging Face model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EmailClassifier:
    """Class to handle email classification"""

    def __init__(self, model_type="traditional", categories=None):
        """
        Initialize email classifier

        Args:
            model_type: Type of model to use ("traditional" or "transformer")
            categories: List of possible email categories
        """
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.categories = categories or [
            "Billing Issues",
            "Technical Support",
            "Account Management",
            "Product Inquiry",
            "General Feedback"
        ]

        if model_type == "transformer":
            # Use pre-trained transformer model
            model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.categories)
            )
        else:
            # Traditional ML model with TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )

            # Create a pipeline with TF-IDF and classifier
            self.model = Pipeline([
                ('tfidf', self.vectorizer),
                ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))
            ])

    def train(self, texts: List[str], labels: List[str]) -> None:
        """
        Train the classification model

        Args:
            texts: List of email texts
            labels: List of corresponding labels
        """
        # Convert string labels to indices
        label_indices = [self.categories.index(label) for label in labels]

        if self.model_type == "transformer":
            # Train the transformer model
            # Note: This is simplified; in practice, you'd use a proper training loop
            # with dataloaders, optimization, etc.
            encoded_texts = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Convert labels to tensor
            labels_tensor = torch.tensor(label_indices)

            # Here we would normally train the model with a proper training loop
            # For simplicity, we'll just use the pre-trained model
            print("Using pre-trained transformer model. Fine-tuning would be implemented here.")

        else:
            # Train the traditional ML model
            self.model.fit(texts, label_indices)

    def predict(self, text: str) -> str:
        """
        Predict category for email text

        Args:
            text: Email text

        Returns:
            str: Predicted category
        """
        if self.model_type == "transformer":
            # Use transformer model for prediction
            encoded_text = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.transformer_model(**encoded_text)
                predictions = outputs.logits
                predicted_idx = torch.argmax(predictions, dim=1).item()

            return self.categories[predicted_idx]
        else:
            # Use traditional ML model for prediction
            predicted_idx = self.model.predict([text])[0]
            return self.categories[predicted_idx]

    def save_model(self, model_path: str = "email_classifier.pkl") -> None:
        """Save the trained model to disk"""
        if self.model_type == "transformer":
            # Save transformer model and tokenizer
            model_dir = "transformer_model"
            os.makedirs(model_dir, exist_ok=True)
            self.transformer_model.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)

            # Save categories
            with open(os.path.join(model_dir, "categories.json"), "w") as f:
                json.dump(self.categories, f)
        else:
            # Save traditional ML model
            with open(model_path, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "categories": self.categories
                }, f)

    @classmethod
    def load_model(cls, model_path: str = "email_classifier.pkl") -> "EmailClassifier":
        """Load a trained model from disk"""
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
            # Load transformer model
            classifier = cls(model_type="transformer")
            classifier.transformer_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            classifier.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Load categories
            with open(os.path.join(model_path, "categories.json"), "r") as f:
                classifier.categories = json.load(f)
        else:
            # Load traditional ML model
            with open(model_path, "rb") as f:
                data = pickle.load(f)

            classifier = cls(model_type="traditional")
            classifier.model = data["model"]
            classifier.categories = data["categories"]

        return classifier


# Function to prepare dummy training data
def prepare_dummy_data() -> Tuple[List[str], List[str]]:
    """
    Create a dummy dataset for email classification

    Returns:
        tuple: (texts, labels)
    """
    # In a real project, you would load actual data here
    # For this assignment, we'll create some dummy examples

    categories = [
        "Billing Issues",
        "Technical Support",
        "Account Management",
        "Product Inquiry",
        "General Feedback"
    ]

    example_emails = [
        # Billing Issues
        "I was charged twice for my last month's subscription. My credit card ending with 1234 shows two payments of $9.99 each on April 15.",
        "I haven't received my refund yet. I canceled my subscription on March 10 and was told I would get a refund within 5-7 business days.",
        "There seems to be a discrepancy in my billing statement. Can you please check my account?",
        "I'm being charged for a premium plan but I only signed up for the basic tier.",
        "My payment failed but I updated my card details. Can you retry the payment?",

        # Technical Support
        "The app keeps crashing whenever I try to upload photos. I'm using an iPhone 13 with the latest iOS.",
        "I can't log into my account. When I enter my password, it just refreshes the page without any error message.",
        "The search functionality is not working on your website. I've tried different browsers.",
        "Videos are not playing correctly on my Android device. They stop after a few seconds.",
        "I'm getting a 404 error when trying to access my account settings page.",

        # Account Management
        "I need to change my email address associated with this account. My current email is user@example.com.",
        "How do I delete my account? I've been trying to find this option in settings.",
        "I want to upgrade my current plan from basic to premium. What are the steps?",
        "Can you help me reset my password? I've forgotten it and can't access my account.",
        "I need to update my shipping address for future deliveries.",

        # Product Inquiry
        "Does your software support integration with Salesforce? We're looking to connect our CRM.",
        "What are the differences between your basic and premium plans?",
        "I'm interested in the enterprise solution. Can you send me more information?",
        "Do you offer bulk discounts for more than 50 licenses?",
        "Is your product compatible with Mac OS Catalina?",

        # General Feedback
        "I love your new update! The interface is much cleaner and easier to navigate.",
        "The customer service I received yesterday was exceptional. The representative was very helpful.",
        "I have a suggestion for improving your checkout process. It currently takes too many steps.",
        "Your latest release has some bugs that need fixing. The notification system isn't working properly.",
        "The new feature you added is exactly what I was looking for. Great job!"
    ]

    labels = []
    for i, _ in enumerate(example_emails):
        category_index = i // 5  # 5 examples per category
        labels.append(categories[category_index])

    return example_emails, labels