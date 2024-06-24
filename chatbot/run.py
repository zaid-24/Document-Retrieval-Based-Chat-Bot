import whisper
model = whisper.load_model("base")
audio = "E:/Intern_Work/LimitPush/chatbot/temp.mp3"
# calculating the time taken 
import time
start = time.time()
result = model.transcribe(audio)
end= time.time()
print(result['text'])
print("time taken to transcribe the audio is:", end-start)
# model = whisper.load_model("base")
# text = model.transcribe("E:/Intern_Work/LimitPush/chatbot/temp.mp3")
# print(text['text'])