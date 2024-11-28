from fastapi import FastAPI, File, UploadFile
import whisper
import uvicorn
import os

app = FastAPI()

# Load the Whisper model
model = whisper.load_model("base")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an audio file to text using Whisper.
    """
    try:
        # Save the uploaded file to disk
        audio_path = f"temp_{file.filename}"
        with open(audio_path, "wb") as buffer:
            buffer.write(await file.read())

        # Transcribe the audio
        result = model.transcribe(audio_path)

        # Clean up the temporary file
        os.remove(audio_path)
        print(f'======================\n{result["text"]}\n=======================')
        return {"text": result["text"]}
    except Exception as e:
        return {"error": str(e)}

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
