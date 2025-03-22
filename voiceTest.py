import ssRecord2 as mr
import pyttsx3
import time
import wave
import pyaudio
import whisper
import os
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer

def speak_text(engine, text):
    """Speak the given text using the provided pyttsx3 engine."""
    engine.say(text)
    engine.runAndWait()

def transcribe_audio(audio_filename):
    """
    Transcribe a WAV file using Whisper.
    The audio file should be 16 kHz, mono.
    """
    result = whisper_model.transcribe(audio_filename, language="en")
    return result["text"]

def generate_response(prompt):
    """
    Generate a response using DialoGPT.
    """
    # Encode the prompt and generate a reply.
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    output_ids = model_llm.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

if __name__ == '__main__': 
    # Initialize two TTS engines
    engine1 = pyttsx3.init()
    engine2 = pyttsx3.init()

    # List available voices and select different ones if available
    voices = engine1.getProperty('voices')
    print("Available voices:")
    for idx, voice in enumerate(voices):
        print(f"{idx}: {voice.name} ({voice.id})")

    # Set voices (adjust indices based on your system)
    engine1.setProperty('voice', voices[0].id)
    if len(voices) > 1:
        engine2.setProperty('voice', voices[1].id)
    else:
        engine2.setProperty('voice', voices[0].id)

    whisper_model = whisper.load_model("base")

    i = 0
    while True:
        print(f"\n--- Conversation Round i ---")
        
        # --- Model 2 listens to Model 1's output ---
        print("Model 2: Listening now (ensure your microphone picks up Model 1's speech)...")
        #audio_file_1 = "recorded_audio2.wav"
        #audio_file_1 = "C:\Class\hack\recorded_audio2.wav"
        #audio_file_1 = r"C:\Class\hack\recorded_audio2.wav"
        # scalar is +/- above average noise for threshold
        # duration in seconds to record to determine average
        # rate in samples per second
        # chunk size is number of samples to use to determine loudness
        audio_file_1 = mr.monitor_and_record(scalar=0.5, duration_for_average=8, rate=16000, chunk_size=16000)
        print(audio_file_1)
        transcribed_text_1 = transcribe_audio(audio_file_1)
        os.remove(audio_file_1)  # Clean up the temporary file.
        print("Model 2 heard:", transcribed_text_1)
        
        # --- Model 2 generates a reply ---
        response_text_2 = generate_response(transcribed_text_1)
        print("Model 2 generated reply:", response_text_2)
        speak_text(engine2, response_text_2)
        i = i + 1