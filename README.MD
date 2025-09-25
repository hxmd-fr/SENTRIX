# SENTRIX - Explain It To Me 💡
SENTRIX is an intelligent, multi-modal web application designed to break down complex topics into simple, easy-to-understand explanations. Using the power of Google's Gemini API and a local zero-shot classification model, this tool can explain concepts from text, images, or even entire webpages. It features a conversational interface, allowing users to ask follow-up questions and dive deeper into any subject.

## Features
- Multi-Modal Input: Get explanations for a typed topic, an uploaded image, or a webpage URL.

- Variable Difficulty: Choose the complexity of the explanation, with levels for a "5-Year-Old," "Teenager," or "College Student."

- Intelligent Topic Classification: A pre-trained Transformer model analyzes text topics to provide better context to the AI, resulting in more relevant explanations.

- Confidence Thresholding: The topic classifier gracefully handles out-of-scope topics by defaulting to a "General" category if its confidence is low.

- Conversational Memory: Ask follow-up questions and the AI will remember the context of the original topic.

- Voice I/O:

  Voice-to-Text: Use your microphone to speak your topic or follow-up question.

- Text-to-Speech: Listen to the AI's explanations read aloud.

- Responsive UI: A clean, modern, and mobile-friendly user interface with a loading spinner and interactive elements.

- Markdown Rendering: Correctly formats the AI's responses, including bold text, bullet points, and numbered lists.

## Tech Stack
This project is built with a modern Python and JavaScript stack.

- Backend:

  Framework: Flask

  AI Service: Google Gemini API (gemini-1.5-flash-latest)

  Local ML Model: Hugging Face transformers with facebook/bart-large-mnli for zero-shot classification.

  Web Scraping: requests & BeautifulSoup4

- Frontend:

  HTML5, CSS3, JavaScript

  Markdown Rendering: Marked.js

  Speech APIs: Web Speech API (for both SpeechRecognition and SpeechSynthesis)

- Environment:

  Python 3.x

  Virtual Environment (venv)

# Setup and Installation
Follow these steps to get the project running on your local machine.

## Clone the Repository:
```
git clone https://github.com/hxmd-fr/SENTRIX.git
cd SENTRIX
```


## Create the venv
```python3 -m venv venv```

## Activate on Windows
```.\venv\Scripts\activate```

## Activate on macOS/Linux
```
source venv/bin/activate
Install Dependencies:
All required packages are listed in requirements.txt. Install them with pip.
```
## Requirements
```
pip install -r requirements.txt
Note: The transformers library will download the pre-trained model files (approx. 1.6 GB) the first time it is run.

Create an Environment File:
Create a file named .env in the root of the project directory. This file will hold your secret API key. Add your Gemini API key to this file:

GEMINI_API_KEY=YOUR_API_KEY_HERE
```
# How to Run
With your virtual environment active and your .env file created, running the application is simple.

```
Run the Flask App:
flask run
```
## Open in Browser

Navigate to http://127.0.0.1:5000 in your web browser to use the application.

