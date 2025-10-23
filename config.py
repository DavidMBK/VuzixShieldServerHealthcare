# config.py

import os
import torch
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# WHISPER CONFIGURATION
# ============================================================================

@dataclass
class WhisperConfig:
    """Configuration for the Whisper model"""
    
    # Model size: "tiny", "base", "small", "medium", "large-v2", "large-v3"
    MODEL_SIZE: str = "large-v2"
    
    # Compute device: "cuda" or "cpu" (auto-detect if left None)
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Compute type: "float16" for CUDA, "int8" for CPU
    COMPUTE_TYPE: str = "float16" if torch.cuda.is_available() else "int8"
    
    # Audio sample rate
    SAMPLE_RATE: int = 16000
    
    # Transcription language
    LANGUAGE: str = None  # Auto-detect language
    
    # Initial context to improve transcription
    INITIAL_CONTEXT: str = "the patient is describing their medical symptoms."
    
    # Number of workers for ThreadPoolExecutor
    MAX_WORKERS: int = 4

# ============================================================================
# SILERO VAD CONFIGURATION
# ============================================================================

@dataclass
class SileroVADConfig:
    """Configuration for Silero Voice Activity Detection"""
    
    # Enable/disable VAD
    ENABLED: bool = True
    
    # Voice activation threshold (0.0-1.0, higher = more restrictive)
    # 0.5 is a good default value, you can lower it to 0.3 to be less restrictive
    THRESHOLD: float = 0.3
    
    # Sample rate required by Silero (DO NOT modify)
    SAMPLE_RATE: int = 16000
    
    # Window size in samples (512 for 16kHz, 256 for 8kHz)
    WINDOW_SIZE_SAMPLES: int = 512
    
    # Minimum padding in milliseconds to consider a segment as speech
    MIN_SPEECH_DURATION_MS: int = 250
    
    # Minimum silence padding in milliseconds to end a segment
    MIN_SILENCE_DURATION_MS: int = 100
    
    # Padding added before and after each voice segment (ms)
    SPEECH_PAD_MS: int = 30
    
    # Minimum audio buffer size before applying VAD (in bytes)
    # Avoids processing buffers that are too small
    MIN_BUFFER_SIZE: int = 8000  # ~0.5 seconds at 16kHz
    
    # Device to run the model
    DEVICE: str = "cpu"  # VAD is lightweight, CPU is sufficient

# ============================================================================
# GEMINI CONFIGURATION
# ============================================================================

@dataclass
class GeminiConfig:
    """Configuration for Google Gemini AI"""
    
    # API Key (loaded from .env)
    API_KEY: str = os.getenv("GEMINI_API_KEY")
    
    # Model name to use
    MODEL_NAME: str = "gemini-2.5-flash"
    
    # Temperature for generation (0.0-1.0, lower = more deterministic)
    TEMPERATURE: float = 0.2
    
    # Request timeout (seconds)
    TIMEOUT: int = 30
    
    # Main system prompt
    SYSTEM_PROMPT: str = """
You are a professional virtual medical assistant. Your task is to assist doctors in collecting information FROM THE PATIENT WHO IS SPEAKING.

**CRITICAL LANGUAGE RULE:**
- ALWAYS look at the patient's MOST RECENT message to determine which language to use
- If the patient's LAST message is in Italian → respond in Italian
- If the patient's LAST message is in English → respond in English
- If the patient's LAST message is in French → respond in French
- The patient may SWITCH languages during the conversation - ALWAYS match their CURRENT language, not their previous language
- ALL JSON fields (problem_summary, medical_summary, clarifying_question) MUST be in the language of the patient's LATEST input

**FUNDAMENTAL DISAMBIGUATION RULE:**
- The PATIENT is ALWAYS the person speaking with you at this moment.
- If the patient mentions other people (e.g., "Luke", "my friend", "my mother"), THOSE ARE NOT THE PATIENT.
- Collect information ONLY about the speaking patient, NOT about third parties mentioned.
- If the patient describes problems of other people, politely ask in THEIR CURRENT language

**PROFESSIONAL CLINICAL FORMAT:**
- The "problem_summary" MUST be written in THIRD PERSON, as in a professional medical record.
- DO NOT use first person → USE third person clinical terms
- The "medical_summary" MUST be in formal clinical language, as a doctor would write.
- Always in the language of the patient's MOST RECENT message

**TRANSFORMATION EXAMPLES FIRST → THIRD PERSON:**

ENGLISH:
- "I have a headache" → problem_summary: "Cephalalgia"
- "my leg hurts" → problem_summary: "Leg pain"
- "I have a high fever" → problem_summary: "High-grade fever"

ITALIAN:
- "ho mal di testa" → problem_summary: "Cefalea"
- "mi fa male la gamba" → problem_summary: "Dolore alla gamba"
- "ho la febbre alta" → problem_summary: "Febbre elevata"

FRENCH:
- "j'ai mal à la tête" → problem_summary: "Céphalée"
- "ma jambe me fait mal" → problem_summary: "Douleur à la jambe"
- "j'ai une forte fièvre" → problem_summary: "Fièvre élevée"

**MEMORY MANAGEMENT RULES:**
1. ALWAYS MAINTAIN the name, age, and problems ALREADY COLLECTED from the speaking patient.
2. If the patient corrects information, use the LAST version stated.
3. If the patient says "not me, but X", problem_summary remains empty or unchanged.
4. Add new information only if it concerns THE SPEAKING PATIENT.
5. ALWAYS convert from first person to clinical third person.
6. **When the patient changes language, keep the existing data but switch your response language immediately**

**REQUIRED JSON OUTPUT:**
{
  "name": "name of THE SPEAKING PATIENT",
  "age": "age of THE SPEAKING PATIENT",
  "problem_summary": "CUMULATIVE summary in CLINICAL THIRD PERSON in the language of patient's LAST message",
  "medical_summary": "professional clinical reformulation in THIRD PERSON in the language of patient's LAST message",
  "clarifying_question": "question in the language of patient's LAST message"
}

**LANGUAGE SWITCHING EXAMPLES:**

SCENARIO: Patient starts in French, then switches to English

Input 1 (French): "Bonjour, je m'appelle Pierre"
Output 1: {
  "name": "Pierre",
  "age": "",
  "problem_summary": "",
  "medical_summary": "Le patient n'a pas encore décrit ses propres symptômes.",
  "clarifying_question": "Pierre, pouvez-vous me décrire VOS symptômes, s'il vous plaît ?"
}

Input 2 (English - LANGUAGE SWITCH): "I have a headache"
Output 2: {
  "name": "Pierre",
  "age": "",
  "problem_summary": "Cephalalgia",
  "medical_summary": "Patient Pierre reports cephalalgia.",
  "clarifying_question": "Pierre, can you describe the type of pain? Is it constant or intermittent?"
}

Input 3 (Italian - ANOTHER LANGUAGE SWITCH): "Ho anche la febbre"
Output 3: {
  "name": "Pierre",
  "age": "",
  "problem_summary": "Cefalea e febbre",
  "medical_summary": "Paziente Pierre riferisce cefalea e febbre.",
  "clarifying_question": "Pierre, da quanto tempo ha la febbre?"
}

**DO NOT make diagnoses. DO NOT suggest calling emergency services. Keep everything brief, clear, in CLINICAL THIRD PERSON, in the patient's CURRENT language (from their LAST message), and focused ON THE PATIENT.**
"""

    
    # Initial confirmation response
    INITIAL_RESPONSE: str = "OK, I understand. I'm ready to collect information from the patient."

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

@dataclass
class ServerConfig:
    """Configuration for the FastAPI server"""
    
    # Server host
    HOST: str = "0.0.0.0"
    
    # Server port
    PORT: int = 8000
    
    # Logging level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    LOG_LEVEL: str = "INFO"
    
    # Log format
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Enable automatic reload (development only)
    RELOAD: bool = False

# ============================================================================
# WEBSOCKET CONFIGURATION
# ============================================================================

@dataclass
class WebSocketConfig:
    """Configuration for WebSocket management"""
    
    # Inactivity timeout before processing audio (seconds)
    INACTIVITY_TIMEOUT: float = 0.5
    
    # Minimum text length to be considered valid
    MIN_TEXT_LENGTH: int = 3

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

@dataclass
class DatabaseConfig:
    """Configuration for MySQL database"""
    
    # Keywords for blood test queries
    BLOOD_TEST_KEYWORDS: list = None
    
    # Keywords for prescription queries
    PRESCRIPTION_KEYWORDS: list = None
    
    # Maximum number of results per query
    MAX_BLOOD_TESTS: int = 3
    MAX_PRESCRIPTIONS: int = 5
    MAX_MEDICAL_HISTORY: int = 3
    
    def __post_init__(self):
        if self.BLOOD_TEST_KEYWORDS is None:
            self.BLOOD_TEST_KEYWORDS = [
                "blood", "test", "exam", "hemoglobin", 
                "glucose", "cholesterol"
            ]
        
        if self.PRESCRIPTION_KEYWORDS is None:
            self.PRESCRIPTION_KEYWORDS = [
                "drug", "medicine", "therapy", "prescription", 
                "pill", "pain"
            ]

# ============================================================================
# MESSAGES CONFIGURATION
# ============================================================================

@dataclass
class MessagesConfig:
    """Default and fallback messages"""
    
    ERROR_TRANSCRIPTION: str = "Error during audio transcription."
    ERROR_GEMINI: str = "Processing error."
    ERROR_DATABASE: str = "Error retrieving data from database."
    ERROR_VAD: str = "Error in Voice Activity Detection."
    
    FALLBACK_QUESTION: str = "Can you repeat?"
    DEFAULT_QUESTION: str = "Can you tell me more?"
    
    EMPTY_BUFFER_WARNING: str = "Empty audio buffer."
    NO_SPEECH_DETECTED: str = "No speech detected (noise only)."

# ============================================================================
# GLOBAL CONFIGURATION INSTANCES
# ============================================================================

whisper_config = WhisperConfig()
silero_vad_config = SileroVADConfig()
gemini_config = GeminiConfig()
server_config = ServerConfig()
websocket_config = WebSocketConfig()
database_config = DatabaseConfig()
messages_config = MessagesConfig()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_whisper_config() -> WhisperConfig:
    """Returns the Whisper configuration"""
    return whisper_config

def get_silero_vad_config() -> SileroVADConfig:
    """Returns the Silero VAD configuration"""
    return silero_vad_config

def get_gemini_config() -> GeminiConfig:
    """Returns the Gemini configuration"""
    return gemini_config

def get_server_config() -> ServerConfig:
    """Returns the Server configuration"""
    return server_config

def get_websocket_config() -> WebSocketConfig:
    """Returns the WebSocket configuration"""
    return websocket_config

def get_database_config() -> DatabaseConfig:
    """Returns the Database configuration"""
    return database_config

def get_messages_config() -> MessagesConfig:
    """Returns the Messages configuration"""
    return messages_config

def validate_config():
    """Validates that all necessary configurations are present"""
    errors = []
    
    if not gemini_config.API_KEY:
        errors.append("GEMINI_API_KEY not found in .env")
    
    if silero_vad_config.ENABLED and silero_vad_config.SAMPLE_RATE != whisper_config.SAMPLE_RATE:
        errors.append(
            f"Silero VAD sample rate ({silero_vad_config.SAMPLE_RATE}) "
            f"must match Whisper ({whisper_config.SAMPLE_RATE})"
        )
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(errors))
    
    return True
