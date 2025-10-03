// /netlify/functions/suggest.js
import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const modelName = "gemini-2.5-flash"; 

// Netlify's main handler function
export async function handler(event, context) {
    if (event.httpMethod !== 'POST') {
        return { statusCode: 405, body: JSON.stringify({ error: 'Method Not Allowed' }) };
    }

    try {
        const body = JSON.parse(event.body);
        const partialTask = body.partial_task;

        if (!partialTask) {
            return { statusCode: 400, body: JSON.stringify({ error: 'Missing partial_task' }) };
        }

        const prompt = `You are an AI task completer. Based on this incomplete text: "${partialTask}", generate 5 diverse and relevant ways to complete the task.`;
        
        const response = await ai.models.generateContent({
            model: modelName,
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: "object",
                    properties: {
                        suggestions: {
                            type: "array",
                            items: { type: "string" },
                            description: "A list of 5 complete task suggestions."
                        }
                    },
                    required: ["suggestions"]
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
        console.error("Gemini Suggestion Function Error:", error);
        return { 
            statusCode: 500, 
            body: JSON.stringify({ error: 'Internal Server Error during suggestion generation.' }) 
        };
    }
}