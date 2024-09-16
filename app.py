import streamlit as st
import pandas as pd
import vosk
import json
import pyaudio
import re
import os
import time
from io import BytesIO

# Paths to the models (adjust paths according to your environment)
model_english_path = r"C:\Users\user\Desktop\vosk\vosk-model-small-en-us-0.15"
model_chinese_path = r"C:\Users\user\Desktop\vosk\vosk-model-small-cn-0.22"

# Function to extract name and country from the transcribed text for English
def extract_info_english(text):
    # Expanded patterns for names
    name_pattern = re.compile(
        r'\b(?:my name is|i am|people know me as|call me|they call me)\s+(\w+)', 
        re.IGNORECASE
    )
    # Expanded patterns for countries/cities
    country_pattern = re.compile(
        r'\b(?:i come from|i belong to|i am from|from)\s+(\w+)', 
        re.IGNORECASE
    )
    
    name_match = name_pattern.search(text)
    country_match = country_pattern.search(text)
    
    name = name_match.group(1) if name_match else "Unknown"
    country = country_match.group(1) if country_match else "Unknown"
    
    return name, country

# Function to extract name and country from the transcribed text for Chinese
def extract_info_chinese(text):
    # Expanded patterns for names (in Chinese)
    name_pattern = re.compile(r'(名字是|我叫|大家叫我|我名叫)\s*(\w+)', re.IGNORECASE)  # My name is / I am called
    # Expanded patterns for countries/cities (in Chinese)
    country_pattern = re.compile(r'(来自|我属于|我从)\s*(\w+)', re.IGNORECASE)  # I come from / I am from
    
    name_match = name_pattern.search(text)
    country_match = country_pattern.search(text)
    
    name = name_match.group(2) if name_match else "Unknown"
    country = country_match.group(2) if country_match else "Unknown"
    
    return name, country

# Function to start the audio stream and recognize speech for a specified duration
def start_stream(duration=5, model=None, extract_func=None):
    global stream, p, rec
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)
    rec = vosk.KaldiRecognizer(model, 16000)
    st.write(f"Listening for {duration} seconds...")
    
    buffer = BytesIO()
    start_time = time.time()
    
    # Listen for the given duration
    while time.time() - start_time < duration:
        data = stream.read(1024)
        buffer.write(data)
        
        if rec.AcceptWaveform(data):
            result = rec.Result()
            result_json = json.loads(result)
            text = result_json.get('text', '')
            
            # Display the transcribed text
            st.write(f"Transcribed Text: {text}")

            # Extract relevant info using the provided extraction function
            name, country = extract_func(text)
            return name, country  # Return the extracted name and country

    return None, None  # If no result is captured

# Function to stop the audio stream
def stop_stream():
    global stream, p
    try:
        if 'stream' in globals() and stream.is_active():
            stream.stop_stream()
            stream.close()
    except NameError:
        st.write("No active stream to stop.")
    finally:
        if 'p' in globals():
            p.terminate()
    st.write("Stopped.")


# Streamlit UI
st.title("Real-Time Speech to Text with Vosk")

# Model selection dropdown
model_choice = st.selectbox('Select Model Language', ['English', 'Chinese'])

# Load the appropriate model based on user selection
if model_choice == 'English':
    model = vosk.Model(model_english_path)
    extract_func = extract_info_english
else:
    model = vosk.Model(model_chinese_path)
    extract_func = extract_info_chinese

# Initialize a DataFrame to hold Name and Country data
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=["Name", "Country"])

# Button to start listening
if st.button('Start Listening'):
    try:
        name, country = start_stream(duration=5, model=model, extract_func=extract_func)  # Listen for 5 seconds
        if name and country:
            # Append the new data to the DataFrame
            new_data = pd.DataFrame([[name, country]], columns=["Name", "Country"])
            st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)

            # Display the DataFrame in a tabular format
            st.table(st.session_state['data'])

            # Save to CSV
            if os.path.exists("data.csv"):
                st.session_state['data'].to_csv("data.csv", mode='a', header=False, index=False)
            else:
                st.session_state['data'].to_csv("data.csv", index=False)
        else:
            st.write("No valid input detected.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        stop_stream()

# Button to stop listening
if st.button('Stop Listening'):
    stop_stream()

# Display the DataFrame in tabular format (always visible)
st.write("Recorded Information:")
st.table(st.session_state['data'])
