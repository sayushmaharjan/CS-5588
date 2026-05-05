import whisper
import json
import warnings
warnings.filterwarnings("ignore")

model = whisper.load_model("tiny")
# We assume there is a narration wav file we can test on
import glob
wavs = glob.glob("outputs/audio/narration/*.wav")
if wavs:
    res = model.transcribe(wavs[0], word_timestamps=True)
    print(json.dumps(res["segments"][0]["words"][:5], indent=2))
else:
    print("No wav files found")
