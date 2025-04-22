# Email Classification System with PII Masking

This project implements an email classification system with PII masking capabilities for a support team. The system can categorize incoming support emails and mask personally identifiable information (PII) before processing.

## Features

- **PII Masking**: Detects and masks personal information such as names, email addresses, phone numbers, etc.
- **Email Classification**: Categorizes emails into different support categories
- **API**: Exposes the functionality through a RESTful API

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/email-classification-system.git
   cd email-classification-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download SpaCy language model:
   ```
   python -m spacy download en_core_web_sm
   ```

### Running the Application

1. Start the API server:
   ```
   python app.py
   ```

   Optional parameters:
   - `--train`: Force training the model before starting the API
   - `--port`: Specify the port number (default: 8000)
   - `--host`: Specify the host (default: 0.0.0.0)

2. The API will be available at `http://localhost:8000`

## API Usage

### Classify Email Endpoint

**URL**: `/classify-email`

**Method**: `POST`

**Request Body**:
```json
{
  "email_body": "Your email content here"
}
```

**Response**:
```json
{
  "input_email_body": "Original email text",
  "list_of_masked_entities": [
    {
      "position": [start_index, end_index],
      "classification": "entity_type",
      "entity": "original_entity_value"
    }
  ],
  "masked_email": "Masked email text",
  "category_of_the_email": "Classified category"
}
```

## Project Structure

- `app.py`: Main application script
- `api.py`: API implementation using FastAPI
- `models.py`: Email classification models
- `utils.py`: Utility functions including PII masking

## Deployment

### Hugging Face Spaces

This application is deployed on Hugging Face Spaces. You can access it at:
[https://huggingface.co/spaces/yourusername/email-classification-system](https://huggingface.co/spaces/yourusername/email-classification-system)

### Local Docker Deployment

1. Build the Docker image:
   ```
   docker build -t email-classification-system .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 email-classification-system
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.