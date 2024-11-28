from fastapi import FastAPI, File, UploadFile
import speech_recognition as sr
from pydub import AudioSegment
import os
import uvicorn

app = FastAPI()

@app.post("/transcribe_sr/")
async def transcribe_audio_sr(file: UploadFile = File(...)):
    """
    Transcribe an audio file to text using SpeechRecognition.
    Converts unsupported formats to WAV before processing.
    """
    try:
        # Save the uploaded file temporarily
        input_audio_path = f"temp_{file.filename}"
        with open(input_audio_path, "wb") as buffer:
            buffer.write(await file.read())

        # Convert to WAV format if necessary
        wav_audio_path = f"{input_audio_path.rsplit('.', 1)[0]}.wav"
        audio = AudioSegment.from_file(input_audio_path)
        audio.export(wav_audio_path, format="wav")
        print(f"Converted {input_audio_path} to {wav_audio_path} successfully. Great job, Sunday!")
        # Load the WAV file with SpeechRecognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_audio_path) as source:
            audio_data = recognizer.record(source)

        # Perform transcription
        text = recognizer.recognize_google(audio_data)

        # Clean up temporary files
        os.remove(input_audio_path)
        os.remove(wav_audio_path)

        return {"text": text}
    except Exception as e:
        return {"error": str(e)}

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
