from gtts import gTTS
from pydub import AudioSegment
import os

TEXT = """
In a world where stories are no longer told by humans,
machines have begun to narrate reality itself.
"""

OUTPUT_FILE = "outputs/narration.wav"

def generate_audio(text, output_path):
    print("[INFO] Generating speech with gTTS...")

    temp_mp3 = "temp.mp3"

    # Generate MP3
    tts = gTTS(text=text, lang="en")
    tts.save(temp_mp3)

    # Convert to WAV
    audio = AudioSegment.from_mp3(temp_mp3)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")

    os.remove(temp_mp3)

    print(f"[DONE] Saved to {output_path}")


if __name__ == "__main__":
    generate_audio(TEXT, OUTPUT_FILE)