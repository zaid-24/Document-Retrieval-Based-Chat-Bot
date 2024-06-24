import whisper
import openai
from gtts import gTTS
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path

app = FastAPI()

# Load Whisper model
model = whisper.load_model("base")

# Define OpenAI API key
openai.api_key = ""

# Set up conversation for ChatCompletion
conversation = [{'role': 'user', 'content': "Message"}]

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded audio file
        file_path = Path("E:/Intern_Work/LimitPush/chatbot/temp.mp3")
        with file_path.open("wb") as audio_file:
            audio_file.write(file.file.read())

        # Load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)

        # Make log-Mel spectrogram and move to the same device as the model.
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Detect the spoken language
        _, probs = model.detect_language(mel)
        translation = model.transcribe(audio, language=max(probs, key=probs.get))

        # Translate the audio using OpenAI API
        file = open("E:/Intern_Work/LimitPush/chatbot/temp.mp3", "rb")
        transcript = openai.Audio.translate("whisper-1", file)

        # Get response from ChatCompletion
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use the appropriate model for your use case
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": conversation},
                ],
                max_tokens=2000
            )

        # Append response to conversation
        conversation.append({
            'role': response.choices[0].message.role,
            'content': response.choices[0].message.content
        })

        # Extract content from the response
        text = response.choices[0].message.content

        # Convert text to speech and save as temp.mp3
        tts = gTTS(text=text, lang='en')
        tts.save("E:/Intern_Work/LimitPush/chatbot/temp2.mp3")

        # Encode the generated audio file to base64
        with open("E:/Intern_Work/LimitPush/chatbot/temp2.mp3", "rb") as f:
            base64_bytes = base64.b64encode(f.read())
            base64_string = base64_bytes.decode('utf-8')

        return {"file_size": len(base64_string), "translation": translation}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Audio file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
