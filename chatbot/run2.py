from openai import OpenAI
client = OpenAI(api_key="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")
# OpenAI.api_key("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")
# client.set_api_key("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")
audio_file= open("E:/Intern_Work/LimitPush/chatbot/temp.mp3", "rb")
import time
start = time.time()
transcript = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file, 
  response_format="text"
)
end= time.time()
print(transcript)
print("time taken to transcribe the audio is:", end-start)