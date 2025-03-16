import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
import tempfile
import os
import wave
from gtts import gTTS  # Google Text-to-Speech
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from huggingface_hub import login
import asyncio  # ✅ Added asyncio import

# ✅ Initialize Asyncio Event Loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))  # This initializes an event loop

# ✅ Set Page Configuration
st.set_page_config(page_title="Echo_Script", page_icon="🎙️", layout="centered")

# ✅ Authenticate with Hugging Face
login("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")

# ✅ Load LLaMA Model
tokenizer = LlamaTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0" , device_map="cpu" )

# ✅ Apply Custom Styling
st.markdown(
    """
    <style>
        /* Light Background */
        .stApp {
            background: linear-gradient(135deg, #F0F8FF, #E6E6FA); /* Light Blue to Lavender Gradient */
            color: black;
        }
        /* Title Styling */
        .stMarkdown h1 {
            color: #4B0082; /* Indigo */
            text-align: center;
            font-size: 2.8em;
            font-weight: bold;
            font-family: 'Arial', sans-serif;
        }
        /* Button Styling */
        .stButton > button {
            background: linear-gradient(90deg, #FFD700, #FFA07A); /* Gold to Light Salmon */
            color: black;
            font-size: 16px;
            font-weight: bold;
            border-radius: 15px;
            padding: 12px 25px;
            border: none;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #FFA07A, #FFD700); /* Reverse Gradient */
            transform: scale(1.05);
        }
        /* File Upload & Input Styling */
        .stFileUploader label, .stTextInput label {
            color: #4B0082; /* Indigo */
            font-size: 18px;
            font-weight: bold;
        }
        /* Transcription Box */
        .stMarkdown p {
            background: rgba(255, 255, 255, 0.8);
            padding: 12px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
        }
        /* Audio Player */
        .stAudio {
            background: rgba(255, 255, 255, 0.5);
            border-radius: 10px;
            padding: 5px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ✅ Function to refine text using LLaMA
def refine_text_with_llama(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    refined_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return refined_text

# ✅ Function to record audio
def record_audio(duration=5, samplerate=44100):
    st.info("🎤 Recording... Speak now!")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()

    temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with wave.open(temp_audio_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())

    return temp_audio_path

# ✅ Function to transcribe audio using Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# ✅ Function to convert text to speech using gTTS
def text_to_speech(text):
    tts = gTTS(text, lang="en")
    temp_speech_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(temp_speech_path)
    return temp_speech_path

# ✅ Streamlit UI
st.title("🎙️ Echo_Script")
st.markdown("Record your voice or upload an audio file to get the transcription and audio output.")

# 🎤 **Record Button**
if st.button("🎤 Record Voice (5s)"):
    audio_path = record_audio()
    st.audio(audio_path, format="audio/wav")

    with st.spinner("⏳ Transcribing..."):
        transcription = transcribe_audio(audio_path)
    
    st.subheader("📝 Transcription:")
    st.markdown(f"<p style='background:rgba(255,255,255,0.8); padding:12px; border-radius:10px;'>{transcription}</p>", unsafe_allow_html=True)

    refined_text = refine_text_with_llama(transcription)
    st.subheader("🔍 Refined Text with LLaMA:")
    st.markdown(f"<p style='background:rgba(255,255,255,0.8); padding:12px; border-radius:10px;'>{refined_text}</p>", unsafe_allow_html=True)

    st.subheader("🔊 Generated Speech:")
    speech_path = text_to_speech(refined_text)
    st.audio(speech_path, format="audio/mp3")

    os.remove(audio_path)
    os.remove(speech_path)

# 📂 **File Uploader**
uploaded_file = st.file_uploader("Or upload an audio file", type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name

    st.audio(uploaded_file, format="audio/wav")

    if st.button("🎧 Transcribe Uploaded Audio"):
        with st.spinner("⏳ Transcribing..."):
            transcription = transcribe_audio(temp_audio_path)
        
        st.subheader("📝 Transcription:")
        st.markdown(f"<p style='background:rgba(255,255,255,0.8); padding:12px; border-radius:10px;'>{transcription}</p>", unsafe_allow_html=True)

        refined_text = refine_text_with_llama(transcription)
        st.subheader("🔍 Refined Text with LLaMA:")
        st.markdown(f"<p style='background:rgba(255,255,255,0.8); padding:12px; border-radius:10px;'>{refined_text}</p>", unsafe_allow_html=True)

        st.subheader("🔊 Generated Speech:")
        speech_path = text_to_speech(refined_text)
        st.audio(speech_path, format="audio/mp3")

        os.remove(temp_audio_path)
        os.remove(speech_path)
