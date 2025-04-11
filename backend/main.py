from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import os
import requests
from typing import List, Dict, Any
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store PDF text in memory (in production, use a proper database)
pdf_texts: Dict[str, str] = {}

@app.get("/")
async def root():
    return {"message": "PDF Question Answering System API is running"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Read PDF content
        pdf_reader = PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # Store the text with filename as key
        pdf_texts[file.filename] = text
        
        return {"message": "PDF uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question")
async def ask_question(question_data: Dict[str, str]):
    if not pdf_texts:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet")
    
    try:
        # Get the first PDF's text (in production, handle multiple PDFs)
        pdf_text = next(iter(pdf_texts.values()))
        
        # Prepare the prompt for Ollama
        prompt = f"""Context from PDF:
{pdf_text}

Question: {question_data['question']}

Please provide a direct answer to the question, including the page numbers where the answer can be found and a brief explanation.

IMPORTANT: Your response must be a valid JSON object with exactly this structure:
{{
    "answer": "your answer here",
    "page_references": [1, 2, 3],
    "explanation": "your explanation here"
}}

Do not include any text before or after the JSON object. The response must be a single, valid JSON object."""
        
        print("Sending prompt to Ollama:", prompt)
        
        # Call Ollama API
        ollama_url = "http://localhost:11434/api/generate"
        ollama_payload = {
            "model": "deepseek-r1:1.5b",
            "prompt": prompt,
            "stream": False
        }
        print("Calling Ollama API with payload:", ollama_payload)
        
        response = requests.post(ollama_url, json=ollama_payload)
        print("Ollama response status:", response.status_code)
        print("Ollama response text:", response.text)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Error calling Ollama API: {response.text}")
        
        # Parse the response
        ollama_response = response.json()
        answer_text = ollama_response.get("response", "").strip()
        print("Raw answer from Ollama:", answer_text)
        
        # Extract JSON from the response
        try:
            # Clean the response text
            cleaned_text = answer_text.replace('\n', ' ').strip()
            
            # Find the JSON object in the response
            start_idx = cleaned_text.find("{")
            end_idx = cleaned_text.rfind("}") + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = cleaned_text[start_idx:end_idx]
            print("Attempting to parse JSON:", json_str)
            
            result = json.loads(json_str)
            print("Parsed JSON result:", result)
            
            # Validate the result structure
            if not all(key in result for key in ["answer", "page_references", "explanation"]):
                raise ValueError("Missing required fields in response")
            
            return result
        except (json.JSONDecodeError, ValueError) as e:
            print("Error parsing JSON:", str(e))
            # If JSON parsing fails, return a structured error
            return {
                "answer": "I apologize, but I encountered an error processing the response.",
                "page_references": [],
                "explanation": str(e)
            }
            
    except Exception as e:
        print("Error in ask_question:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 