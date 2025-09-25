import { SpeedInsights } from "@vercel/speed-insights/next"
import os
from flask import Flask, render_template, request, session # NEW: import session
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import requests
from bs4 import BeautifulSoup

from classifier import classify_topic

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create a Flask web application
app = Flask(__name__)

app.secret_key = 'your_super_secret_key'


# Create the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash-latest')

def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs])
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    # NEW: Initialize or clear chat history at the start
    if 'chat_history' not in session or request.method == 'GET':
        session['chat_history'] = []

    if request.method == 'POST':
        # Check if it's a new topic or a follow-up
        if 'new_topic_form' in request.form:
            session.clear() # Clear history for a new topic
            session['chat_history'] = []
            
            topic = request.form.get('topic')
            image_file = request.files.get('image_file')
            url = request.form.get('url')
            difficulty = request.form.get('difficulty')
            
            initial_prompt = ""
            user_message = ""
            
            if url:
                webpage_text = get_text_from_url(url)
                if webpage_text:
                    user_message = f"Explain this article about '{url}'"
                    initial_prompt = f"Explain the key points of the following article to me like I'm a {difficulty}: {webpage_text[:4000]}"
                else:
                    return render_template('index.html', error="Sorry, I couldn't read the content from that URL.")
            elif image_file and image_file.filename != '':
                img = Image.open(image_file)
                user_message = "Explain this image."
                initial_prompt = [f"Explain the concept in this image to me like I'm a {difficulty}.", img]
            elif topic:
                user_message = f"Explain '{topic}'"
                candidate_labels = ["Science", "History", "Technology", "Art", "Health", "Finance"]
                predicted_label, score = classify_topic(topic, candidate_labels)
                final_category = predicted_label if score > 0.5 else "General"
                initial_prompt = f"Explain the {final_category} topic '{topic}' to me like I'm a {difficulty}."
            else:
                return render_template('index.html', error="Please submit a URL, an image, or a topic.")

            # Start a new chat session with the model
            chat = model.start_chat(history=[])
            response = chat.send_message(initial_prompt)
            
            session['chat_history'].append({'role': 'user', 'parts': [user_message]})
            session['chat_history'].append({'role': 'model', 'parts': [response.text]})

        elif 'follow_up_form' in request.form:
            follow_up_prompt = request.form.get('follow_up')
            
            # Recreate chat object with existing history
            chat = model.start_chat(history=session.get('chat_history', []))
            response = chat.send_message(follow_up_prompt)
            
            session['chat_history'].append({'role': 'user', 'parts': [follow_up_prompt]})
            session['chat_history'].append({'role': 'model', 'parts': [response.text]})
        
        # This is needed to make the session changes save
        session.modified = True
            
    return render_template('index.html', chat_history=session.get('chat_history', []))

if __name__ == '__main__':
    app.run(debug=True)