from transformers import pipeline
import librosa
import os
# next 3 lines to overcome some issues due to using a mac silicon chip
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ensure no GPU fallback
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

def detect_emotion(audio_path):
    # loading pretrained emotion recognition pipeline
    classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
    
    # load and resample audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # running the model
    results = classifier(y)

    # sort and get the top emotion
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    highest_confidence_emotion = results[0]['label']
    confidence = results[0]['score']
    
    return results, highest_confidence_emotion, confidence

def interpret_emotion_detection(highest_confidence_emotion, confidence):
    if highest_confidence_emotion == "neutral":
        return f"Your tone sounds calm and professional ({confidence:.0%} confidence)."
    elif highest_confidence_emotion == "happy":
        return f"You sound enthusiastic — great energy! ({confidence:.0%} confidence)."
    elif highest_confidence_emotion == "sad":
        return f"You sound slightly low or hesitant — try adding more energy. ({confidence:.0%} confidence)"
    elif highest_confidence_emotion == "angry":
        return f"Your tone sounds tense — focus on keeping your voice relaxed. ({confidence:.0%} confidence)"
    else:
        return f"Detected tone: {highest_confidence_emotion} ({confidence:.0%} confidence)."