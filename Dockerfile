FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Pre-train the model
RUN python -c "from models import EmailClassifier, prepare_dummy_data; texts, labels = prepare_dummy_data(); classifier = EmailClassifier(model_type='traditional'); classifier.train(texts, labels); classifier.save_model('email_classifier.pkl')"

# Expose port for API
EXPOSE 7860

# Start the application
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]