# app.py (Final Code with HTML Serving and Proxy Fix)

import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory # Added send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import List

# --- FIX: Corrected Imports from langchain_core ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser 
from werkzeug.middleware.proxy_fix import ProxyFix
# --- END FIX ---

# Load environment variables (for local testing; Render loads securely)
load_dotenv() 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# --- Pydantic Schemas for Structured Output ---

class TaskAnalysis(BaseModel):
    text: str = Field(description="The cleaned-up final task text.")
    time: str = Field(description="Extracted deadline or date. Must be YYYY-MM-DD or a time of day/unspecified.")
    category: str = Field(description="A category label.")
    urgent: bool = Field(description="True if the task is marked urgent.")
    note: str = Field(description="A helpful, concise note.")
    effort_score: str = Field(description="Assigned score: Low, Medium, High, or Critical.", 
                              enum=["Low", "Medium", "High", "Critical"])

class SuggestionList(BaseModel):
    suggestions: List[str] = Field(description="A list of 5 complete task suggestions.")


# --- INITIALIZE FLASK AND GEMINI ---
app = Flask(__name__)

# Apply ProxyFix immediately to handle Render's reverse proxy headers
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1) 

CORS(app) 

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=GEMINI_API_KEY
)

def get_today_string():
    return datetime.now().strftime("%Y-%m-%d")


# --- ROUTE 1: Serves the Frontend HTML ---
@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def serve_frontend(path):
    # This route serves the frontend HTML file (index.html) from the static folder
    return send_from_directory('static', path)


# --- ROUTE 2: /api/analyze (Task Analysis) ---
@app.route("/api/analyze", methods=["POST"])
def analyze_handler():
    # ... (Analysis logic remains the same) ...
    try:
        data = request.get_json()
        task_text = data.get("task_text")
        current_date_string = get_today_string()

        if not task_text:
            return jsonify({"error": "Missing task_text"}), 400

        parser = JsonOutputParser(pydantic_object=TaskAnalysis)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             f"You are a professional task analysis engine. The current date is {current_date_string}. "
             f"Your sole purpose is to return ONLY a valid JSON object. "
             f"Strictly format any date found as YYYY-MM-DD. \n"
             f"{parser.get_format_instructions()}"
            ),
            ("user", task_text),
        ])
        
        chain = prompt | llm | parser
        result = chain.invoke({})

        return jsonify(result)

    except Exception as e:
        print(f"ERROR in analyze.py: {e}")
        return jsonify({"error": "Internal Server Error during AI analysis."}), 500


# --- ROUTE 3: /api/suggest (Dynamic Suggestions) ---
@app.route("/api/suggest", methods=["POST"])
def suggest_handler():
    # ... (Suggestion logic remains the same) ...
    try:
        data = request.get_json()
        partial_task = data.get("partial_task")

        if not partial_task:
            return jsonify({"error": "Missing partial_task"}), 400

        parser = JsonOutputParser(pydantic_object=SuggestionList)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a helpful AI completer. Generate 5 unique suggestions to complete the user's partial text. "
             f"The response format must be:\n{parser.get_format_instructions()}"
            ),
            ("user", partial_task),
        ])
        
        chain = prompt | llm | parser
        result = chain.invoke({})

        return jsonify(result)

    except Exception as e:
        print(f"ERROR in suggest.py: {e}")
        return jsonify({"error": "Internal Server Error during suggestion generation."}), 500


# --- STARTUP COMMAND FOR RENDER ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
