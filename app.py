# /netlify/functions/main.py

import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

# LangChain Imports for Structured Output
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import JsonOutputParser

# Load environment variables for local testing (Netlify loads them automatically in production)
load_dotenv() 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # This is a security check.
    raise ValueError("GEMINI_API_KEY environment variable not found. Please set it in Netlify or .env.")

# --- INITIAL SETUP ---
app = Flask(__name__)
# Enable CORS for the frontend's domain (e.g., Netlify URL)
CORS(app) 

# Initialize the Gemini Model (using LangChain wrapper)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=GEMINI_API_KEY
)

# --- Pydantic Schemas for Structured Output ---

class TaskAnalysis(BaseModel):
    """Schema for structured output of task details."""
    text: str = Field(description="The cleaned-up final task text.")
    time: str = Field(description="Extracted deadline or date. Must be YYYY-MM-DD or a time of day/unspecified.")
    category: str = Field(description="A category label (e.g., Work, Study, Health).")
    urgent: bool = Field(description="True if the task is marked urgent or uses words like ASAP.")
    note: str = Field(description="A helpful, concise note or warning for the user.")
    effort_score: str = Field(description="Assigned score: Low, Medium, High, or Critical.", 
                              enum=["Low", "Medium", "High", "Critical"])

class SuggestionList(BaseModel):
    """Schema for structured output of suggested tasks."""
    suggestions: List[str] = Field(description="A list of 5 complete task suggestions.")


# Helper to get today's date
def get_today_string():
    return datetime.now().strftime("%Y-%m-%d")

# --- HANDLERS FOR FRONTEND ROUTES ---

# 1. /api/analyze route (Task Analysis)
@app.route("/api/analyze", methods=["POST"])
def analyze_handler():
    try:
        data = request.get_json()
        task_text = data.get("task_text")
        current_date_string = get_today_string()

        if not task_text:
            return jsonify({"error": "Missing task_text"}), 400

        # LangChain Setup for Structured Analysis
        parser = JsonOutputParser(pydantic_object=TaskAnalysis)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             f"You are a professional task analysis engine. The current date is {current_date_string}. "
             f"Your sole purpose is to analyze the user's task and return ONLY a valid JSON object matching the schema. "
             f"Strictly format any date found as YYYY-MM-DD. "
             f"The response format must be:\n{parser.get_format_instructions()}"
            ),
            ("user", task_text),
        ])
        
        chain = prompt | llm | parser
        
        # Invoke the LLM
        result = chain.invoke({})

        return jsonify(result)

    except ValidationError as e:
        print(f"Validation Error: {e}")
        return jsonify({"error": "AI returned invalid structure.", "details": str(e)}), 500
    except Exception as e:
        print(f"ERROR in analyze.py: {e}")
        return jsonify({"error": "Internal Server Error during AI analysis."}), 500


# 2. /api/suggest route (Dynamic Suggestions)
@app.route("/api/suggest", methods=["POST"])
def suggest_handler():
    try:
        data = request.get_json()
        partial_task = data.get("partial_task")

        if not partial_task:
            return jsonify({"error": "Missing partial_task"}), 400

        # LangChain Setup for Suggestions
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


# --- Deployment Entry Point (Gunicorn/Serverless) ---
# When deploying to Netlify, Vercel, or Render, they use an adapter (like gunicorn)
# that finds the 'app' variable (your Flask app) and runs it. 
# This structure is sufficient for the deployment toolchain.
if __name__ == '__main__':
    # This only runs if you run 'python main.py' locally.
    # In production, gunicorn or the serverless wrapper executes the 'app'.
    app.run(debug=True)