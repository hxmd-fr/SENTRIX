import os
import io
import re
import logging
from flask import Flask, render_template, request, Response, session, send_file
from flask_session import Session
from dotenv import load_dotenv
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from fpdf import FPDF

# --- SETUP ---
# Configure structured logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from your .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Create and configure the Flask app
app = Flask(__name__)
# Configure server-side sessions to handle large chat histories
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Initialize the Gemini Model
model = genai.GenerativeModel('gemini-1.5-flash-latest')


# --- HELPER FUNCTIONS ---
def search_google(query):
    """Performs a Google search and returns the top 3 result links."""
    try:
        params = {"q": query, "api_key": SERPAPI_API_KEY}
        search = GoogleSearch(params)
        results = search.get_dict()
        return [r['link'] for r in results.get('organic_results', [])[:3]]
    except Exception as e:
        logging.error(f"Error during Google Search: {e}")
        return []

def scrape_text_from_url(url):
    """Fetches and extracts paragraph text from a webpage."""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs])
    except Exception as e:
        logging.error(f"Error fetching or parsing URL {url}: {e}")
        return ""

def clean_markdown_for_fpdf(text):
    """Removes or replaces Markdown for a cleaner plain-text PDF."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\* (.*?)\n', r'- \1\n', text)
    text = re.sub(r'#+\s*(.*?)\n', r'\1\n\n', text)
    return text


# --- FLASK ROUTES ---
@app.route('/')
def index():
    """Renders the main page and clears any previous session history."""
    session.clear()
    return render_template('index.html', chat_history=session.get('chat_history', []))

@app.route('/generate', methods=['POST'])
def generate():
    """Handles the initial research goal and generates the main report."""
    session.clear()
    session['chat_history'] = []
    research_goal = request.form.get('research_goal')
    
    logging.info(f"Received new research goal: {research_goal}")
    urls = search_google(research_goal)
    scraped_content = ""
    for url in urls:
        scraped_content += scrape_text_from_url(url) + "\n\n"
    
    synthesis_prompt = f"Based on the following articles, write a detailed report on the topic: '{research_goal}'. Use Markdown. ARTICLES: --- {scraped_content[:8000]}"
    
    try:
        logging.info("Synthesizing report...")
        response = model.generate_content(synthesis_prompt)
        full_response_text = response.text

        session['chat_history'].append({'role': 'user', 'parts': [research_goal]})
        session['chat_history'].append({'role': 'model', 'parts': [full_response_text]})
        session.modified = True
    except Exception as e:
        logging.error(f"Error during generation: {e}")
        full_response_text = "An error occurred while generating the report."

    def stream_text(text):
        yield text

    return Response(stream_text(full_response_text), mimetype='text/plain')

@app.route('/follow-up', methods=['POST'])
def follow_up():
    """Handles follow-up questions using the existing chat history."""
    follow_up_prompt = request.form.get('follow_up')
    logging.info(f"Received follow-up: {follow_up_prompt}")
    
    session['chat_history'].append({'role': 'user', 'parts': [follow_up_prompt]})
    
    try:
        chat = model.start_chat(history=session['chat_history'])
        response = chat.send_message(follow_up_prompt)
        full_response_text = response.text

        session['chat_history'].append({'role': 'model', 'parts': [full_response_text]})
        session.modified = True
    except Exception as e:
        logging.error(f"Error during follow-up: {e}")
        full_response_text = "An error occurred during the follow-up."

    def stream_text(text):
        yield text

    return Response(stream_text(full_response_text), mimetype='text/plain')

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    """Generates a PDF from plain text and sends it for download."""
    data = request.get_json()
    report_text = data.get('text', '')
    logging.info("Generating PDF...")

    clean_text = clean_markdown_for_fpdf(report_text)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.multi_cell(0, 10, txt=clean_text.encode('latin-1', 'replace').decode('latin-1'))
    
    pdf_bytes = pdf.output(dest='S')
    
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name='ai_report.pdf'
    )

@app.route('/download-report')
def download_report():
    """Serves the last model response from the session as a text file."""
    try:
        chat_history = session.get('chat_history', [])
        last_model_response = ""
        for message in reversed(chat_history):
            if message['role'] == 'model':
                last_model_response = message['parts'][0]
                break
        
        if not last_model_response:
            return "No report found in session.", 404
        
        logging.info("Downloading report as .md file.")
        return Response(
            last_model_response,
            mimetype="text/markdown",
            headers={"Content-disposition": "attachment; filename=ai_report.md"}
        )
    except Exception as e:
        logging.error(f"Error during report download: {e}")
        return "Could not download the report.", 500

if __name__ == '__main__':
    app.run(debug=True)
