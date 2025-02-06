import streamlit as st
import whisper
import openai
import io
from dotenv import load_dotenv
import os
from pydub import AudioSegment
import tempfile
from openai import OpenAI
from gtts import gTTS

load_dotenv()

# Get the OpenAI API key from the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to convert text to speech and play it
def text_to_speech(text: str, lang="en"):
    tts = gTTS(text=text, lang=lang)
    
    # Save the speech to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio_file:
        tts.save(tmp_audio_file.name)
        st.audio(tmp_audio_file.name, format="audio/mp3", autoplay=True)

# Initialize Whisper model
whisper_model = whisper.load_model("base")

# Function to transcribe speech to text using Whisper
def transcribe_audio(audio_file):
    # Convert the uploaded file to a format Whisper can understand (wav)
    audio = AudioSegment.from_file(audio_file)
    
    # Save the audio as a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        audio.export(tmp_file, format="wav")
        tmp_file_path = tmp_file.name

    # Load the audio into Whisper
    audio_array = whisper.load_audio(tmp_file_path)
    
    # Transcribe the audio
    result = whisper_model.transcribe(audio_array)
    
    # Cleanup: Delete the temporary file after processing
    os.remove(tmp_file_path)
    
    return result['text']

# Function to query OpenAI GPT-4 for medical advice
# Function to query OpenAI GPT-4 for medical advice
def get_medical_advice(symptoms_text):
    # Create OpenAI client instance
    client = OpenAI(api_key=openai.api_key)

    # Prepare the messages for the chat model
    messages = [
        {"role": "system", "content": "You are an expert Doctor. Answer shortly to the patient's query. resolve patient queries"},
        {"role": "user", "content": f"User describes the following symptoms: {symptoms_text}. Provide a general recommendation and give medicine prescrpotions."}
    ]
    
    # Request medical advice from the chat model
    response = client.chat.completions.create(
        model="gpt-4",  # You can also use "gpt-3.5-turbo" or another model
        messages=messages
    )
    
    # Correctly extract and return the medical advice from the response
    return response.choices[0].message.content.strip()


# Streamlit app layout
st.logo(
    "logo.png",
    size="medium",
    link="https://github.com/talibraath",
)

st.title("Cure Yourself")
st.write("Welcome to Cure Yourself! Record a voice message describing your symptoms, and we'll transcribe it and provide you with medical advice.") 
audio_value = st.audio_input("Record a voice message to transcribe", key="audio_input_1")
# Transcribe the audio
if audio_value:
    text = transcribe_audio(audio_value)
    st.write("Audio Text: ", text)
    # Get medical advice from GPT-4
    advice = get_medical_advice(text)
    st.write("Medical Advice: ", advice)
    # Text to speech for medical advice
    text_to_speech(advice)