# 🤖SENTRIX - AI Research Assistant
This project is a web-based AI agent that automates the research process. Given a single research goal, the agent autonomously searches the web for relevant sources, scrapes their content, and synthesizes the information into a comprehensive, well-formatted report.

## Key Features
Autonomous Research: Provide a high-level goal, and the agent handles the entire research workflow.

Dynamic Source-Finding: Uses the SerpApi service to perform real-time Google searches for the most relevant and up-to-date articles.

Web Content Extraction: Automatically scrapes and parses the text content from multiple source URLs.

AI-Powered Synthesis: Leverages the Google Gemini API to read the collected information and generate a cohesive, structured report.

Markdown Formatting: The final report is formatted with Markdown for clear headings, lists, and structure.

## Tech Stack
### Backend
Framework: Flask

AI Service: Google Gemini API (gemini-1.5-flash-latest)

Web Search: SerpApi (google-search-results)

Web Scraping: requests & BeautifulSoup4

### Frontend
Structure: HTML5, CSS3

Interactivity: Vanilla JavaScript

Markdown Rendering: Marked.js

## Getting Started
Follow these steps to get the project running on your local machine.

### Prerequisites
```
Python 3.8+

A Google Gemini API Key.

A SerpApi API Key.
```

### Installation
Clone the Repository
```
git clone https://github.com/hxmd-fr/SENTRIX
cd SENTRIX
```
### Create and Activate a Virtual Environment

```python3 -m venv venv```

### Activate on Windows
```.\venv\Scripts\activate```

### Activate on macOS/Linux
```source venv/bin/activate```

## Install Dependencies

```pip install -r requirements.txt``` <br>
Set Up Environment File <br>
Create a file named .env in the root of the project and add your API keys:
```
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
SERPAPI_API_KEY=YOUR_SERPAPI_API_KEY_HERE
```
## How to Run
With your virtual environment active, start the Flask development server:
```flask run``` <br>
The application will be available at http://127.0.0.1:5000.