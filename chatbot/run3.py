import torchaudio
import speechbrain as sb
from speechbrain.inference.ASR import EncoderASR

asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-14-en", savedir="pretrained_models/asr-wav2vec2-commonvoice-14-en")
asr_model.transcribe_file("speechbrain/asr-wav2vec2-commonvoice-14-en/example_en.wav")
print(asr_model.transcribe_file('E:/Intern_Work/LimitPush/chatbot/temp.mp3'))