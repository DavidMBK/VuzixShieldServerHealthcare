# main.py

import numpy as np
import torch
import asyncio
import json
import logging
from typing import Dict, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from contextlib import asynccontextmanager
from starlette.websockets import WebSocketState

from google.genai import Client
from google.genai.types import GenerateContentConfig

# Silero VAD
torch.set_num_threads(1)  # Optimize PyTorch for single-threaded inference

# Import MySQL database
from database import init_db, get_db, Patient, BloodTest, Prescription, MedicalHistory
from sqlalchemy.orm import Session

# Import configurations
from config import (
    whisper_config,
    silero_vad_config,
    gemini_config,
    server_config,
    websocket_config,
    database_config,
    messages_config,
    validate_config
)

# --- Config Logging ---
logging.basicConfig(
    level=getattr(logging, server_config.LOG_LEVEL),
    format=server_config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Global variables 
model = None
vad_model = None
genai_client = Client(api_key=gemini_config.API_KEY)
whisper_executor = ThreadPoolExecutor(max_workers=whisper_config.MAX_WORKERS)

def load_silero_vad():
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        model.to(silero_vad_config.DEVICE)
        logger.info(f"Silero VAD LOADED ON {silero_vad_config.DEVICE}")
        return model, utils
    except Exception as e:
        logger.error(f"Error loading Silero VAD: {e}")
        return None, None

# VAD function check if speech is present in audio or not
# Returns True if speech is detected, False otherwise
def detect_speech(audio_np: np.ndarray, vad_model, threshold: float = 0.5) -> bool:
    
    if vad_model is None or not silero_vad_config.ENABLED:
        return True   # If VAD model is not loaded or disabled, assume speech present
    
    try:
        # Convert audio to torch tensor
        audio_tensor = torch.from_numpy(audio_np).float()
        
        # Silero VAD needs audio on 16kHz
        if len(audio_tensor) < silero_vad_config.WINDOW_SIZE_SAMPLES:
            return False
        
        # Divide audio into chunks
        num_samples = len(audio_tensor)
        num_chunks = num_samples // silero_vad_config.WINDOW_SIZE_SAMPLES
        
        speech_probs = []
        
        for i in range(num_chunks):
            start_idx = i * silero_vad_config.WINDOW_SIZE_SAMPLES
            end_idx = start_idx + silero_vad_config.WINDOW_SIZE_SAMPLES
            chunk = audio_tensor[start_idx:end_idx]
            
            # Calculate probability of speech in the chunk
            with torch.no_grad():
                speech_prob = vad_model(chunk, silero_vad_config.SAMPLE_RATE).item()
                speech_probs.append(speech_prob)
        
        if not speech_probs:
            return False
        
        # Calculate average speech probability
        avg_speech_prob = sum(speech_probs) / len(speech_probs)
        
        # Log for debugging
        logger.info(f"🎤 VAD: Speech Probability = {avg_speech_prob:.3f} (threshold: {threshold})")
        
        # Return True if average probability exceeds threshold
        return avg_speech_prob >= threshold
        
    except Exception as e:
        logger.error(f"{messages_config.ERROR_VAD}: {e}")
        return True  # In case of error, assume speech is present

# WebSocket and FastAPI related code
@dataclass
class ClientSession:
    websocket: WebSocket
    audio_buffer: bytearray = field(default_factory=bytearray)
    whisper_context: str = field(default_factory=lambda: whisper_config.INITIAL_CONTEXT)
    is_in_speech_event: bool = False
    inactivity_task: asyncio.Task = None
    
    conversation_history: List[Dict] = field(default_factory=list)
    patient_data: Dict = field(default_factory=lambda: {
        "name": "",
        "age": "",
        "problem_summary": "",
        "medical_summary": ""
    })
    
    def reset_buffer(self):
        self.audio_buffer.clear()
        self.is_in_speech_event = False
    
    # Add message to conversation history
    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "parts": [{"text": content}]
        })
    
    # Update patient data from Gemini response
    def update_patient_data(self, gemini_response: dict):
        if gemini_response.get("name"):
            self.patient_data["name"] = gemini_response["name"]
        if gemini_response.get("age"):
            self.patient_data["age"] = gemini_response["age"]
        if gemini_response.get("problem_summary"):
            self.patient_data["problem_summary"] = gemini_response["problem_summary"]
        if gemini_response.get("medical_summary"):
            self.patient_data["medical_summary"] = gemini_response["medical_summary"]

# Manages active WebSocket connections
class ConnectionManager:
    
    def __init__(self):
        self.active_sessions: Dict[WebSocket, ClientSession] = {}
    
    # Connect a new client and create a session
    async def connect(self, websocket: WebSocket) -> ClientSession:
        await websocket.accept()
        session = ClientSession(websocket=websocket)
        
        # Initialize conversation history with system prompt and initial response
        session.add_to_history("user", gemini_config.SYSTEM_PROMPT)
        session.add_to_history("model", gemini_config.INITIAL_RESPONSE)
        
        self.active_sessions[websocket] = session
        logger.info(f"🔗 Client connected. Total active clients: {len(self.active_sessions)}")
        return session
    
    # Disconnect a client and clean up session
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_sessions:
            session = self.active_sessions[websocket]
            if session.inactivity_task and not session.inactivity_task.done():
                session.inactivity_task.cancel()
            del self.active_sessions[websocket]
            logger.info(f"🔌 Client disconnected. Active clients remaining: {len(self.active_sessions)}")
    
    def get_session(self, websocket: WebSocket) -> ClientSession:
        return self.active_sessions.get(websocket)

manager = ConnectionManager()

# Lifespan event to manage startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vad_model
    
    # Validate configurations
    try:
        validate_config()
        logger.info("Configurations successfully validated")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    
    # Load Silero VAD
    if silero_vad_config.ENABLED:
        logger.info("Loading Silero VAD...")
        vad_model, _ = load_silero_vad()
    
    # Load Whisper
    logger.info(
        f"Loading Whisper model '{whisper_config.MODEL_SIZE}' "
        f"on device '{whisper_config.DEVICE}'..."
    )
    model = WhisperModel(
        whisper_config.MODEL_SIZE,
        device=whisper_config.DEVICE,
        compute_type=whisper_config.COMPUTE_TYPE
    )
    
    # Initialize MySQL database
    logger.info("🗄️ Initializing MySQL database...")
    init_db()
    logger.info("MySQL database ready!")
    
    logger.info("Whisper model loaded. Thread pool ready.")
    
    yield
    
    # Cleanup
    whisper_executor.shutdown(wait=True)
    logger.info("Application terminated.")

app = FastAPI(lifespan=lifespan)

def transcribe_sync(audio_np: np.ndarray, context: str) -> str:
    """
    Transcribes the audio using Whisper synchronously.
    This function is executed in the ThreadPoolExecutor.
    """
    segments, _ = model.transcribe(
        audio_np,
        language=whisper_config.LANGUAGE,
        initial_prompt=context,
    )
    
    result = ""
    for segment in segments:
        result += segment.text.strip() + " "
    
    return result.strip()

def get_relevant_medical_data(patient_name: str, problem_keywords: str, db: Session) -> dict:
    """
    Retrieves relevant medical data from the MySQL database based on the patient's name
    and problem keywords.
    """
    result = {
        "patient_info": None,
        "blood_tests": [],
        "prescriptions": [],
        "medical_history": []
    }
    
    if not patient_name:
        return result
    
    try:
        # Patient info
        patient = db.query(Patient).filter(Patient.name.ilike(f"%{patient_name}%")).first()
        if patient:
            result["patient_info"] = {
                "name": patient.name,
                "age": patient.age,
                "blood_type": patient.blood_type,
                "allergies": patient.allergies
            }
        
        keywords_lower = problem_keywords.lower()
        
        # Blood tests
        if any(word in keywords_lower for word in database_config.BLOOD_TEST_KEYWORDS):
            blood_tests = db.query(BloodTest).filter(
                BloodTest.patient_name.ilike(f"%{patient_name}%")
            ).order_by(BloodTest.test_date.desc()).limit(database_config.MAX_BLOOD_TESTS).all()
            
            result["blood_tests"] = [{
                "date": str(test.test_date),
                "hemoglobin": test.hemoglobin,
                "white_blood_cells": test.white_blood_cells,
                "platelets": test.platelets,
                "glucose": test.glucose,
                "cholesterol": test.cholesterol,
                "notes": test.notes
            } for test in blood_tests]
        
        # Prescriptions
        if any(word in keywords_lower for word in database_config.PRESCRIPTION_KEYWORDS):
            prescriptions = db.query(Prescription).filter(
                Prescription.patient_name.ilike(f"%{patient_name}%")
            ).order_by(Prescription.start_date.desc()).limit(database_config.MAX_PRESCRIPTIONS).all()
            
            result["prescriptions"] = [{
                "medication": p.medication,
                "dosage": p.dosage,
                "frequency": p.frequency,
                "start_date": str(p.start_date),
                "notes": p.notes
            } for p in prescriptions]
        
        # Medical history (always included)
        history = db.query(MedicalHistory).filter(
            MedicalHistory.patient_name.ilike(f"%{patient_name}%")
        ).order_by(MedicalHistory.diagnosed_date.desc()).limit(database_config.MAX_MEDICAL_HISTORY).all()
        
        result["medical_history"] = [{
            "condition": h.condition,
            "diagnosed_date": str(h.diagnosed_date),
            "status": h.status,
            "notes": h.notes
        } for h in history]
        
    except Exception as e:
        logger.error(f"{messages_config.ERROR_DATABASE}: {e}")
    
    return result

async def process_final_audio(session: ClientSession):
    """
    Processes the final audio when speech ends.
    Performs VAD, transcription, and requests the response from Gemini.
    """
    if not session.audio_buffer:
        logger.warning(messages_config.EMPTY_BUFFER_WARNING)
        session.is_in_speech_event = False
        return
    
    # Check minimum buffer size
    if len(session.audio_buffer) < silero_vad_config.MIN_BUFFER_SIZE:
        logger.info("⏭️ Buffer too small, skipping.")
        session.reset_buffer()
        return
    
    try:
        # Convert audio to numpy format
        audio_np = np.frombuffer(session.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        
        # CHECK WITH VAD IF THERE IS SPEECH
        if silero_vad_config.ENABLED:
            has_speech = detect_speech(audio_np, vad_model, silero_vad_config.THRESHOLD)
            
            if not has_speech:
                logger.info(f"🔇 {messages_config.NO_SPEECH_DETECTED}")
                session.reset_buffer()
                
                # Send notification to client
                await session.websocket.send_text(json.dumps({
                    "type": "no_speech",
                    "message": "Only noise detected, no speech."
                }))
                return
        
        # IF THERE IS SPEECH, TRANSCRIBE
        loop = asyncio.get_event_loop()
        final_text_str = await loop.run_in_executor(
            whisper_executor,
            transcribe_sync,
            audio_np,
            session.whisper_context
        )
        
        session.reset_buffer()
        
        # Process only if the text is significant
        if final_text_str and len(final_text_str) > websocket_config.MIN_TEXT_LENGTH:
            logger.info(f"Transcription: '{final_text_str}'")
            
            # Add to history
            session.add_to_history("user", final_text_str)
            
            # Send transcription to client
            await session.websocket.send_text(json.dumps({
                "type": "final_transcript",
                "text": final_text_str
            }))
            
            # Request response from Gemini
            logger.info("🤖 Requesting from Gemini...")
            try:
                gemini_response = await get_gemini_response(session)
                session.update_patient_data(gemini_response)
                session.add_to_history("model", json.dumps(gemini_response))
                
                # Retrieve medical data from database
                db = next(get_db())
                medical_data = get_relevant_medical_data(
                    session.patient_data["name"],
                    session.patient_data["problem_summary"],
                    db
                )
                
                # Send complete response to client
                await session.websocket.send_text(json.dumps({
                    "type": "final_response",
                    "data": {
                        "name": session.patient_data["name"],
                        "age": session.patient_data["age"],
                        "problem_summary": session.patient_data["problem_summary"],
                        "medical_summary": session.patient_data["medical_summary"],
                        "clarifying_question": gemini_response.get(
                            "clarifying_question",
                            messages_config.DEFAULT_QUESTION
                        ),
                        "medical_data": medical_data
                    }
                }))
            except Exception as e:
                logger.error(f"Gemini error: {e}")
                
    except Exception as e:
        logger.error(f"{messages_config.ERROR_TRANSCRIPTION}: {e}")
        session.is_in_speech_event = False

# Obtain a structured response from Gemini based on conversation history
async def get_gemini_response(session: ClientSession) -> dict:
    try:
        response = await genai_client.aio.models.generate_content(
            model=gemini_config.MODEL_NAME,
            contents=session.conversation_history,
            config=GenerateContentConfig(
                temperature=gemini_config.TEMPERATURE,
                response_mime_type="application/json"
            )
        )

        response_json = json.loads(response.text)
        logger.info(f"Gemini response: {response_json}")
        return response_json
        
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        # Fallback response
        return {
            "name": session.patient_data.get("name", ""),
            "age": session.patient_data.get("age", ""),
            "problem_summary": session.patient_data.get("problem_summary", ""),
            "medical_summary": messages_config.ERROR_GEMINI,
            "clarifying_question": messages_config.FALLBACK_QUESTION
        }

# WebSocket endpoint for transcription
@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    
    session = await manager.connect(websocket)
    
    try:
        while True:
            # Check if WebSocket is still connected
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            
            try:
                message = await websocket.receive()
            except RuntimeError as e:
                if "disconnect message has been received" in str(e):
                    break
                raise
            
            # Handle audio data (bytes)
            if "bytes" in message:
                session.audio_buffer.extend(bytearray(message["bytes"]))
                
                # Skip transcription if already in speech event
                if session.is_in_speech_event:
                    continue

                # Partial transcription
                # Can be removed if not needed for low-latency applications
                try:
                    if len(session.audio_buffer) >= silero_vad_config.MIN_BUFFER_SIZE:
                        audio_np = np.frombuffer(
                            session.audio_buffer,
                            dtype=np.int16
                        ).astype(np.float32) / 32768.0
                        
                        # Check VAD before transcription
                        if silero_vad_config.ENABLED:
                            has_speech = detect_speech(audio_np, vad_model, silero_vad_config.THRESHOLD)
                            if not has_speech:
                                continue  # Skip if no speech detected
                        
                        loop = asyncio.get_event_loop()
                        partial_text = await loop.run_in_executor(
                            whisper_executor,
                            transcribe_sync,
                            audio_np,
                            session.whisper_context
                        )
                            
                        if partial_text:
                            await websocket.send_text(json.dumps({
                                "type": "partial_transcript",
                                "text": partial_text
                            }))
                        
                except Exception as e:
                    logger.error(f"Partial transcription error: {e}")

            # Handle text data (speech_end event)
            elif "text" in message:
                data = json.loads(message["text"])
                
                if data.get("event") == "speech_end":
                    logger.info("ℹ️ 'speech_end' signal received.")
                    
                    # Delete timer task if exists
                    if session.inactivity_task and not session.inactivity_task.done():
                        session.inactivity_task.cancel()
                    
                    # Create a new inactivity timer
                    session.inactivity_task = asyncio.create_task(
                        asyncio.sleep(websocket_config.INACTIVITY_TIMEOUT)
                    )
                    
                    def on_timer_done(task):
                        if not task.cancelled():
                            asyncio.create_task(process_final_audio(session))
                    
                    session.inactivity_task.add_done_callback(on_timer_done)

    except WebSocketDisconnect:
        logger.info("🔌 Client disconnected.")
    except Exception as e:
        logger.error(f"💥 WebSocket error: {e}", exc_info=True)
    finally:
        manager.disconnect(websocket)

# Root endpoint for live check
@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Medical Assistant API v1.0",
        "whisper_model": whisper_config.MODEL_SIZE,
        "device": whisper_config.DEVICE,
        "vad_enabled": silero_vad_config.ENABLED,
        "vad_threshold": silero_vad_config.THRESHOLD
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": len(manager.active_sessions),
        "model_loaded": model is not None,
        "vad_loaded": vad_model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=server_config.HOST,
        port=server_config.PORT,
        log_level=server_config.LOG_LEVEL.lower(),
        reload=server_config.RELOAD
    )
