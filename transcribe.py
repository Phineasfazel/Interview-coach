from openai import OpenAI
import os
import whisper

# uses openai whisper to get transcription of audio
def transcribe_audio(audio_path):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            #response_format="verbose_json"
        )
    
    print(transcript)

    return transcript.text

# def time_before_first_word(video_path):
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     with open(video_path, "rb") as video_file:
#         transcript = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=video_file,
#             response_format="verbose_json"  # important if you want timestamps!
#         )

#     # transcript.text gives you the full text
#     print("Transcript:", transcript.text)

#     # If verbose_json is used, segments include timestamps:
#     first_segment = transcript.segments[0]
#     print("First word starts at:", first_segment["start"], "seconds")


    # model = whisper.load_model("base")  # or "small" / "medium"
    # result = model.transcribe(video_path)
    
    # # Whisper returns segments with start times
    # if len(result["segments"]) == 0:
    #     return None  # No speech detected
    
    # first_segment = result["segments"][0]
    # last_segment = result["segments"][-1]
    # print(first_segment)
    # print(last_segment)
    # #return first_segment["start"]