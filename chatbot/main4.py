from fastapi import FastAPI, File, UploadFile, HTTPException
import openai
from bs4 import BeautifulSoup
import requests
from whisperspeech.pipeline import Pipeline
import whisper
import numpy as np
from io import BytesIO
from pydub import AudioSegment
import pyttsx3  # Import pyttsx3 library

app = FastAPI()

# Set your OpenAI API key
openai.api_key = ""

# Initialize the Whisper Speech pipeline
whisper_pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')

# Function to search a website for the answer using BeautifulSoup
def search_website(query, url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Implement your logic to extract relevant information from the website
    # For simplicity, let's assume the answer is in the text of the paragraphs
    paragraphs = soup.find_all('p')
    for paragraph in paragraphs:
        if query.lower() in paragraph.text.lower():
            return paragraph.text
    return None

# Function to get an answer from ChatGPT
def chat_gpt(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the appropriate model for your use case
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            max_tokens=2000
        )
        if response is None or response.get("choices") is None:
            raise HTTPException(status_code=500, detail="OpenAI response is empty")
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error with ChatGPT: {e}")
        return None

# Function to convert text to voice using pyttsx3
def text_to_voice(text, output_path):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_path)
    engine.runAndWait()

# Function to transcribe audio using Whisper Live AI
def transcribe_audio(audio_file):
    try:
        model = whisper.load_model("base")  # Load the Whisper model
        transcript = model.transcribe(audio_file)["text"]
        return transcript
    except Exception as e:
        print(f"Error with Whisper transcription: {e}")
        return None
    
# API endpoint to answer questions
@app.get("/answer/{question}")
def get_answer(question: str):
    # Define the websites to search
    websites = ["https://www.ready-able.com/10-common-plumbing-repair-questions-answered/"]

    # Search each website for the answer
    for website in websites:
        website_answer = search_website(question, website)
        if website_answer:
            # Convert the answer to voice
            voice_response_path = 'E:/Intern_Work/LimitPush/chatbot/website_generated.wav'
            text_to_voice(website_answer, voice_response_path)
            return {"source": website, "answer": website_answer, "voice_response_path": voice_response_path}

    # If the answer is not found in the websites, ask ChatGPT
    gpt_answer = chat_gpt(question)
    # Convert the answer to voice
    voice_response_path = 'E:/Intern_Work/LimitPush/chatbot/gpt_generated.wav'
    text_to_voice(gpt_answer, voice_response_path)
    return {"source": "ChatGPT", "answer": gpt_answer, "voice_response_path": voice_response_path}

# API endpoint to handle MP3 input and answer questions
@app.post("/answer/mp3")
async def answer_mp3(audio_file: UploadFile = File(...)):
    try:
        audio_content = await audio_file.read()  # Read the audio file content

        # Convert MP3 bytes to AudioSegment
        audio_segment = AudioSegment.from_mp3(BytesIO(audio_content))

        # Convert AudioSegment to NumPy array with dtype=float32
        audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

        # Normalize the audio array to floating-point values between -1 and 1
        audio_array = audio_array / np.max(np.abs(audio_array), axis=0)

        # Transcribe the audio
        audio = "E:/Intern_Work/LimitPush/chatbot/temp.mp3" 
        model = whisper.load_model("base")
        result = model.transcribe(audio)
        question = result['text']
        print(question)

        # If transcription is successful, proceed with answering
        if question:
            answer = chat_gpt(question)
            # ... (your existing logic for answering questions using websites or ChatGPT)
            voice_response_path = 'E:/Intern_Work/LimitPush/chatbot/mp3_generated.wav'
            text_to_voice(answer, voice_response_path)  # Convert the answer to voice
            return {"source": "ChatGPT", "answer": answer, "voice_response_path": voice_response_path}
        else:
            return {"error": "Transcription failed"}

    except Exception as e:
        print(f"Error processing MP3 file: {e}")
        return {"error": "Internal server error"}
