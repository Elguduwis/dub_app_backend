from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import whisper
import os
import uuid
import shutil
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment

app = FastAPI(title="Dub App Backend")

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model - Switched to "tiny" to fit in 512MB RAM
print("Loading Whisper tiny model...")
model = whisper.load_model("tiny")
print("Backend ready!")

@app.get("/")
async def root():
    return {"message": "Dub App Backend is running", "status": "online"}

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    """Process uploaded video/audio and return transcription"""
    
    # Generate unique ID for this file
    file_id = str(uuid.uuid4())
    original_ext = os.path.splitext(file.filename)[1]
    original_path = f"/tmp/{file_id}{original_ext}"
    
    # Save uploaded file
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Convert to audio (WAV format)
        audio_path = f"/tmp/{file_id}.wav"
        
        if original_ext.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
            # Extract audio from video
            video = VideoFileClip(original_path)
            video.audio.write_audiofile(audio_path, codec='pcm_s16le')
            video.close()
        else:
            # Convert audio to WAV
            audio = AudioSegment.from_file(original_path)
            audio.export(audio_path, format="wav")
        
        # Transcribe with Whisper (fp16=False prevents warnings on CPU-only servers)
        result = model.transcribe(audio_path, word_timestamps=True, fp16=False)
        
        # Format segments
        segments = []
        for segment in result["segments"]:
            segments.append({
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "text": segment["text"].strip()
            })
        
        # Clean up
        os.remove(original_path)
        os.remove(audio_path)
        
        return JSONResponse({
            "success": True,
            "segments": segments,
            "duration": result["segments"][-1]["end"] if segments else 0,
            "message": f"Successfully transcribed {len(segments)} segments"
        })
        
    except Exception as e:
        # Clean up on error
        for path in [original_path, audio_path]:
            if os.path.exists(path):
                os.remove(path)
        
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": "Processing failed"
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
