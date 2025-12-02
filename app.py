import gc
import os
import json
import base64 # Required for decoding the image
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import List

# --- Core AI and Flask Dependencies ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser 
from werkzeug.middleware.proxy_fix import ProxyFix
from google import genai
from google.genai import types
# --- END Dependencies ---

# Load environment variables (Render loads securely from dashboard)
load_dotenv() 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in Render.")

# --- Pydantic Schemas for Structured Output ---

class TaskAnalysis(BaseModel):
    """Schema for structured output of task details."""
    text: str = Field(description="The cleaned-up final task text.")
    time: str = Field(description="Extracted deadline or date. Must be YYYY-MM-DD or a time of day/unspecified.")
    category: str = Field(description="A category label (e.g., Work, Study, Health).")
    urgent: bool = Field(description="True if the task is marked urgent or uses words like ASAP.")
    note: str = Field(description="A helpful, concise note or warning for the user.")
    effort_score: str = Field(description="Assigned score: Low, Medium, High, or Critical.", 
                              enum=["Low", "Medium", "High", "Critical", "Unspecified"])

class SuggestionList(BaseModel):
    """Schema for returning suggestions."""
    suggestions: List[str] = Field(description="A list of 5 complete task suggestions.")

class ExtractedTasks(BaseModel):
    """Schema for returning multiple structured tasks extracted from an image."""
    extracted_tasks: List[TaskAnalysis] = Field(description="A list of structured tasks extracted from the image.")


# --- INITIALIZE FLASK AND GEMINI ---
app = Flask(__name__, static_folder='static') 

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1) 
CORS(app) 
client = genai.Client(api_key=GEMINI_API_KEY)

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
    return send_from_directory(app.static_folder, path)


# --- ROUTE 2: /api/analyze (Text Analysis) ---
@app.route("/api/analyze", methods=["POST"])
def analyze_handler():
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
             "Your sole purpose is to return ONLY a valid JSON object. "
             "Strictly format any date found as YYYY-MM-DD. \n"
             "{format_instructions}"
            ),
            ("user", "{user_input}"),
        ])
        
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())
        chain = prompt | llm | parser
        
        result = chain.invoke({"user_input": task_text})

        return jsonify(result)

    except Exception as e:
        print(f"ERROR in analyze.py: {e}")
        return jsonify({"error": f"Internal Server Error during AI analysis: {str(e)}"}), 500


# --- ROUTE 3: /api/suggest (Dynamic Suggestions) ---
@app.route("/api/suggest", methods=["POST"])
def suggest_handler():
    try:
        data = request.get_json()
        partial_task = data.get("partial_task")

        if not partial_task:
            return jsonify({"error": "Missing partial_task"}), 400

        parser = JsonOutputParser(pydantic_object=SuggestionList)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a helpful AI completer. Generate 5 unique suggestions to complete the user's partial text. "
             f"The response format must be:\n"
             "{format_instructions}"
            ),
            ("user", "{user_input}"),
        ])
        
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())

        chain = prompt | llm | parser
        
        result = chain.invoke({"user_input": partial_task})

        return jsonify(result)

    except Exception as e:
        print(f"ERROR in suggest.py: {e}")
        return jsonify({"error": f"Internal Server Error during suggestion generation: {str(e)}"}), 500


# --- ROUTE 4: /api/capture (Multimodal Image Analysis) ---
@app.route("/api/capture", methods=["POST"])
def capture_handler():
    try:
        data = request.get_json()
        image_b64 = data.get("image_data")
        text_prompt = data.get("text_prompt")

        if not image_b64 or not text_prompt:
            return jsonify({"error": "Missing image or prompt data"}), 400
            
        # --- FIX: Robust Base64 Decoding ---
        # 1. Split the data URI string (e.g., "data:image/jpeg;base64,iVBORw...")
        image_data_uri_parts = image_b64.split(',')
        
        # 2. Extract MIME type (e.g., image/jpeg)
        mime_type = image_data_uri_parts[0].split(':')[1].split(';')[0]
        
        # 3. Decode ONLY the Base64 string (the second part of the split)
        image_bytes = base64.b64decode(image_data_uri_parts[1])

        gc.collect()
        # 4. Create the Multimodal Part (Image)
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type 
        )
        
        # 5. Setup LangChain Chain for structured output
        parser = JsonOutputParser(pydantic_object=ExtractedTasks)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert OCR and task extraction agent. Analyze the provided image and text prompt. Return all extracted tasks in the required JSON format. \n{format_instructions}"),
            ("user", [image_part, text_prompt]) # Multimodal Input: Image and Text together
        ])
        
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())
        chain = prompt | llm | parser
        
        # 6. Invoke the LLM
        result = chain.invoke({})

        return jsonify(result)

    except Exception as e:
        # Catch decoding and API errors here and return a clear 500
        print(f"ERROR in capture.py: {e}")
        return jsonify({"error": f"Internal Server Error during image processing: {str(e)}"}), 500


# --- STARTUP COMMAND FOR RENDER (Gunicorn) ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


