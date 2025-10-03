// /netlify/functions/analyze.js
import { GoogleGenAI } from '@google/genai';

// --- CONFIGURATION ---
// Netlify securely injects the GEMINI_API_KEY environment variable.
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const modelName = "gemini-2.5-flash"; 

// Helper to get today's date in YYYY-MM-DD format for the AI prompt
const getTodayString = () => {
    const d = new Date();
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
};

// Netlify's main handler function
export async function handler(event, context) {
    if (event.httpMethod !== 'POST') {
        return { statusCode: 405, body: JSON.stringify({ error: 'Method Not Allowed' }) };
    }

    try {
        const body = JSON.parse(event.body);
        const taskText = body.task_text;
        const currentDateString = getTodayString();

        if (!taskText) {
            return { statusCode: 400, body: JSON.stringify({ error: 'Missing task_text' }) };
        }

        // --- PROMPT for Structured Analysis (Includes Date Fix) ---
        const prompt = `CURRENT DATE: ${currentDateString}. Analyze the following user task: "${taskText}". Extract the task details, category, and urgency. For the 'time' field, format any specific date strictly as YYYY-MM-DD relative to the CURRENT DATE, or use 'unspecified' or the time of day. Crucially, analyze the complexity and time investment required and assign an effort_score. Generate a helpful, concise note for the user.`;
        
        const response = await ai.models.generateContent({
            model: modelName,
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: "object",
                    properties: {
                        text: { type: "string" }, 
                        time: { type: "string", description: "Date in YYYY-MM-DD or time of day." }, 
                        category: { type: "string", description: "Category label." },
                        urgent: { type: "boolean" }, 
                        note: { type: "string", description: "Helpful, concise note." },
                        effort_score: { 
                            type: "string", 
                            enum: ["Low", "Medium", "High", "Critical"], 
                            description: "Complexity score." 
                        }
                    },
                    required: ["text", "time", "category", "urgent", "note", "effort_score"]
                },
            }
        });

        const result = JSON.parse(response.text.trim());

        return {
            statusCode: 200,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(result)
        };

    } catch (error) {
        console.error("Gemini Analyze Function Error:", error);
        return { 
            statusCode: 500, 
            body: JSON.stringify({ error: 'Internal Server Error during AI analysis.' }) 
        };
    }
}