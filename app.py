import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

# --- Core AI and Flask Dependencies ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage # <--- Critical Import for Images
from werkzeug.middleware.proxy_fix import ProxyFix

# --- Load Environment Variables ---
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in Render or .env file.")

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

class RetrospectiveAnalysis(BaseModel):
    """Schema for the productivity coaching report."""
    summary: str = Field(description="A 1-sentence summary of performance (e.g., 'You crushed the low-hanging fruit but avoided the big project').")
    productivity_score: int = Field(description="A score from 0-100 based on completion rate and effort.")
    patterns: List[str] = Field(description="3 specific observations about the user's behavior (e.g., 'You procrastinate on High effort tasks').")
    advice: List[str] = Field(description="3 actionable coaching tips for next week.")
    
# --- INITIALIZE FLASK AND GEMINI ---
app = Flask(__name__, static_folder='static')

# Fix for running behind a proxy (like Render/Heroku)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)
CORS(app)

# Initialize LangChain Gemini Chat Model
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


# --- ROUTE 4: /api/capture (Multimodal Image Analysis - FIXED) ---
@app.route("/api/capture", methods=["POST"])
def capture_handler():
    try:
        data = request.get_json()
        image_b64 = data.get("image_data")
        text_prompt = data.get("text_prompt")

        if not image_b64 or not text_prompt:
            return jsonify({"error": "Missing image or prompt data"}), 400
            
        # 1. Split the data URI string (e.g., "data:image/jpeg;base64,iVBORw...")
        # We split only once on the comma to separate header from data
        header, encoded = image_b64.split(",", 1)
        
        # Extract mime type (e.g., "image/jpeg")
        mime_type = header.split(":")[1].split(";")[0]

        # 2. Setup Parser for JSON output
        parser = JsonOutputParser(pydantic_object=ExtractedTasks)
        
        # 3. Construct the message using LangChain's HumanMessage.
        # This prevents the "Invalid Template" error by handling the image structure natively.
        full_prompt_text = (
            f"You are an expert OCR and task extraction agent. {text_prompt}\n"
            f"INSTRUCTIONS:\n"
            f"1. Transcribe ALL visible text, including HANDWRITING.\n"
            f"2. Extract every single item that looks like a task, reminder, list item, or note.\n"
            f"3. If a line of text is ambiguous, include it as a task anyway with category 'General'.\n"
            f"4. If there is visible text but no clear 'tasks', create ONE task containing a summary of the text.\n"
            f"5. NEVER return an empty list if there is text in the image.\n"
            f"Return the result ONLY as valid JSON. Do not include markdown ticks.\n"
            f"{parser.get_format_instructions()}"
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": full_prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"}
                }
            ]
        )
        
        # 4. Invoke the chain directly with the message list
        # We bypass ChatPromptTemplate to inject the image payload directly
        chain = llm | parser
        
        result = chain.invoke([message])

        return jsonify(result)

    except Exception as e:
        # Catch decoding and API errors here and return a clear 500
        print(f"ERROR in capture.py: {e}")
        return jsonify({"error": f"Internal Server Error during image processing: {str(e)}"}), 500
        
# --- ROUTE 5: /api/retrospective (Productivity Coaching) ---
@app.route("/api/retrospective", methods=["POST"])
def retrospective_handler():
    try:
        data = request.get_json()
        completed_tasks = data.get("completed", [])
        pending_tasks = data.get("pending", [])

        # Validate we have data to analyze
        if not completed_tasks and not pending_tasks:
             return jsonify({"error": "No tasks found to analyze."}), 400

        parser = JsonOutputParser(pydantic_object=RetrospectiveAnalysis)
        
        # 1. Convert lists to string format
        completed_str = json.dumps(completed_tasks)
        pending_str = json.dumps(pending_tasks)

        # 2. Define prompt with placeholders {completed} and {pending}
        # We DO NOT put the JSON directly here, or LangChain will crash.
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an elite Productivity Coach. Analyze the user's 'Completed' vs 'Pending' tasks. "
             "Look for patterns: Are they ignoring high-effort tasks? Are they completing urgent items? "
             "Be constructive but honest. "
             "The response format must be:\n"
             "{format_instructions}"
            ),
            ("user", "Here is my data:\n\nCOMPLETED TASKS:\n{completed}\n\nPENDING TASKS:\n{pending}"),
        ])
        
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())
        
        chain = prompt | llm | parser
        
        # 3. Pass the actual JSON strings here safely
        result = chain.invoke({
            "completed": completed_str,
            "pending": pending_str
        })

        return jsonify(result)

    except Exception as e:
        print(f"ERROR in retrospective.py: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# --- STARTUP COMMAND FOR RENDER (Gunicorn) ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


