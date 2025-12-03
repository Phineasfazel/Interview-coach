import os
import subprocess
import tempfile
import streamlit as st
from openai import OpenAI
from transcribe import transcribe_audio
from analysis import generate_feedback
from emotiondetection import detect_emotion, interpret_emotion_detection
import json

st.set_page_config(page_title = "AI Interview Coach - Video Upload")

st.title("👔 AI Interview Coach")

question = st.text_input("What interview question were you answering?")
role = st.text_input("What role are you applying for?")
additional_context = st.text_input("Any additional context?")

# Create output directory:
# gets current working directory
# adds an outputs subfolder to it if it doesnt already exist
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#shows an upload button on the web page
uploaded = st.file_uploader("Upload the video of your interview (.mp4 or .mov)", type=["mp4", "mov"], accept_multiple_files=False)

# function that uses ffmpeg to convert uploaded video to a mono .wav file
def extract_audio(video_path, out_wav_path):
    #Extract mono 16 kHz audio from a video using ffmpeg
    cmd = ["ffmpeg","-y","-i", video_path,"-ac", "1","-ar", "16000","-vn",out_wav_path]    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_video_time(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])

def return_wpm(time_spoken_seconds, transcript):
    transcript_list = transcript.split(' ')
    count = 0
    for word in transcript_list:
        count += 1
    time_spoken_minutes = time_spoken_seconds/60
    wpm = count/time_spoken_minutes
    return wpm

def interpret_wpm(wpm):
    if wpm > 165:
        return f'Your speaking speed is too fast: your wpm are {wpm}.'
    elif wpm < 90:
        return f'Your speaking speed is too slow: your wpm are {wpm}.'
    else:
        return f'Your speaking speed is good: your wpm are {wpm}.'

transcript_exists = False

# Use session state to store transcript so we don't re-transcribe each time
try:
    x = st.session_state.transcript
except:
    st.session_state.transcript = None

if uploaded:
    # create a temporary directory prefixed by upload to decrease file clutter
    tempdir = tempfile.mkdtemp(prefix="upload_")
    # extract the file name eg mov or mp4
    ext = os.path.splitext(uploaded.name)[1].lower()

    video_file_name = "input_"+uploaded.name+ext
    # combine name and file type to create a filename
    video_path = os.path.join(tempdir, video_file_name)
    # opens a new file and writes video bytes from file to it
    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())

    audio_name = "audio_" + uploaded.name + ".wav"
    # creates an audio file in chosen folder with picked name 
    audio_path = os.path.join(OUTPUT_DIR, audio_name)
    
    #call earlier function to do ffmpeg conversion
    extract_audio(video_path, audio_path)
    # show success message
    st.success("Audio succesfully saved! ✅")

    #time_before_first_word(video_path)

    if audio_path:
        # checking if a transcript already exists so we dont have to do this again
        if st.session_state.transcript == None:
            with st.spinner("Transcribing video ..."):
                st.session_state.transcript = transcribe_audio(audio_path)
        st.success("Transcription complete! ✅")
        transcript = transcribe_audio(audio_path)

        time_spoken_seconds = get_video_time(video_path)
        wpm = return_wpm(time_spoken_seconds, transcript)

        # providing feedback to user part
        st.divider()
        st.subheader("Interview Feedback")
        
        feedback_type = st.radio(
        "Select feedback type:",
        ("Short", "Detailed"))
    
        if st.session_state.transcript.strip():
            if st.button("Generate Feedback"):
                # calling the function in the analysis file to send the transcript to an nlp model
                with st.spinner("Analysing your answer ..."):
                    feedback = generate_feedback(st.session_state.transcript, question, role, additional_context, feedback_type)
                    # getting emotion feedback
                    results, highest_confidence_emotion, confidence = detect_emotion(audio_path)
                    emotion_feedback = interpret_emotion_detection(highest_confidence_emotion, confidence)

                # showing feedback
                st.subheader(f"{feedback_type} Feedback")
                st.markdown(feedback)

                st.subheader("Tone and Emotion Feedback")
                st.write(emotion_feedback)

                st.subheader('Delivery feedback')
                st.write(interpret_wpm(wpm))

# future ideas:
# pronounciation/articulation feedback (helpful for non native speakers)

# import whisper
# model = whisper.load_model("base")
# result = model.transcribe("audio.wav")
# segments = result["segments"]

# delivery score - wpm, pauses, filler words/min

# def get_video_duration(video_path):
#     cmd = [
#         "ffprobe", "-v", "error",
#         "-show_entries", "format=duration",
#         "-of", "json", video_path
#     ]
#     result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     info = json.loads(result.stdout)
#     return float(info["format"]["duration"])

# pitch/volume variation (somwhat covered by emotion detection but a useful addition)
# Body language feedback (using mediapipe pose / face detection)

# streamlit run app.py