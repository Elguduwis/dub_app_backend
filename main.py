import os
# FORCE PyTorch to use minimal memory before it even loads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import whisper
import uuid
import shutil
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
import torch

# Force CPU mode for PyTorch
torch.set_num_threads(1)

app = FastAPI(title="Dub App Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Whisper tiny model on CPU...")
# Explicitly force CPU device to prevent any CUDA memory allocation
model = whisper.load_model("tiny", device="cpu")
print("Backend ready!")

@app.get("/")
async def root():
    return {"message": "Dub App Backend is running", "status": "online"}

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    original_ext = os.path.splitext(file.filename)[1]
    original_path = f"/tmp/{file_id}{original_ext}"
    
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        audio_path = f"/tmp/{file_id}.wav"
        
        if original_ext.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
            video = VideoFileClip(original_path)
            video.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
            video.close()
        else:
            audio = AudioSegment.from_file(original_path)
            audio.export(audio_path, format="wav")
        
        # fp16=False is critical for CPU execution
        result = model.transcribe(audio_path, word_timestamps=True, fp16=False)
        
        segments = []
        for segment in result["segments"]:
            segments.append({
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "text": segment["text"].strip()
            })
        
        os.remove(original_path)
        os.remove(audio_path)
        
        return JSONResponse({
            "success": True,
            "segments": segments,
            "duration": result["segments"][-1]["end"] if segments else 0,
            "message": f"Successfully transcribed {len(segments)} segments"
        })
        
    except Exception as e:
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
