# 🏥 Medical Assistant Voice Recognition Server

<div align="center">

A powerful real-time voice-activated medical assistant server that combines cutting-edge speech recognition with AI-powered clinical insights. Built for healthcare professionals who need fast, accurate, and intelligent voice transcription.

[![Python 3.11.2](https://img.shields.io/badge/Python-3.11.2-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![MySQL 8.0+](https://img.shields.io/badge/MySQL-8.0+-orange.svg)](https://www.mysql.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## ✨ Features

### 🎤 Advanced Speech Processing
- **Real-time Speech Recognition** powered by Faster-Whisper (large-v2 model)
- **Voice Activity Detection (VAD)** using Silero VAD for intelligent noise filtering
- **GPU Acceleration** with CUDA support for lightning-fast transcription
- **Multi-language Support** with automatic detection (English, Italian, French, and more)

### 🤖 AI-Powered Intelligence
- **Clinical Language Processing** that converts conversational speech into professional medical records
- **Smart Responses** using Google Gemini 2.5 Flash for contextual medical questions
- **Context-Aware** understanding of medical terminology and clinical scenarios

### 💾 Database Integration
- **MySQL Backend** for secure storage of patient records
- **Complete Medical Records** including blood tests, prescriptions, and medical history
- **WebSocket API** for low-latency real-time communication

---

## 📋 Prerequisites

### System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.11.2 |
| MySQL | 8.0 or higher |
| GPU | NVIDIA GPU (recommended) |
| CUDA Toolkit | 12.1 |
| Operating System | Windows |

### API Keys

You'll need a **Google Gemini API Key** to use the AI features. Get your free API key at [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key).

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

git clone <https://github.com/DavidMBK/VuzixShieldServerHealthcare>
cd VuzixShieldServerHealthcare


### 2️⃣ Verify Python Installation

python --version

Expected output: `Python 3.11.2`

If Python is not installed, download it from [python.org](https://www.python.org/).

### 3️⃣ Create Virtual Environment

#### Create virtual environment: 

python -m venv .venv

#### Activate virtual environment (Windows PowerShell):

.venv\Scripts\Activate.ps1

**Note**: If you encounter an execution policy error, run:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

### 4️⃣ Install Dependencies

#### Upgrade pip:

pip install --upgrade pip

#### Install all required packages: 

pip install -r requirements.txt


**Installation time**: The PyTorch CUDA installation may take 5-10 minutes.

### 5️⃣ Install CUDA Toolkit

1. Download and install CUDA 12.1 from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
2. Follow this [installation guide](https://www.youtube.com/watch?v=4wPUtUtSp-o)
3. Verify installation: nvcc --version

#### Expected output:

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Feb__8_05:53:42_Coordinated_Universal_Time_2023
Cuda compilation tools, release 12.1, V12.1.66
Build cuda_12.1.r12.1/compiler.32415258_0


---

## 🗄️ Database Setup

### Install MySQL

1. Download MySQL 8.0+ from [mysql.com](https://www.mysql.com/)
2. Follow this [installation guide](https://www.youtube.com/watch?v=AaISTiooIVU)
3. Verify installation: mysql --version

#### Expected output: `mysql  Ver 8.0.42 for Win64 on x86_64 (MySQL Community Server - GPL)`

### Create Database

#### Connect to MySQL and create the database:

CREATE DATABASE medical_db;


---

## ⚙️ Configuration

### Environment Variables

**⚠️ CRITICAL**: Create a `.env` file in the project root directory with your credentials.

===== GEMINI AI CONFIGURATION =====

GEMINI_API_KEY=your_gemini_api_key_here

===== MYSQL DATABASE CONFIGURATION =====

MYSQL_HOST=localhost

MYSQL_PORT=3306

MYSQL_USER=root

MYSQL_PASSWORD=your_mysql_password_here

MYSQL_DATABASE=medical_db

**Replace these values:**
- `your_gemini_api_key_here`: Get from [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)
- `your_mysql_password_here`: Your MySQL root password
- `3306`: Default MySQL port (change if different)

**Security Note**: The `.env` file contains sensitive credentials and should never be committed to version control.

---

## 🎛️ Advanced Configuration

The `config.py` file contains application behavior settings. Customize these based on your needs:

### Whisper Settings

MODEL_SIZE: str = "large-v2" # Options: tiny, base, small, medium, large-v2, large-v3

LANGUAGE: str = None # None = auto-detect, or specify "en", "it", "fr"

DEVICE: str = "cuda" # "cuda" for GPU, "cpu" for CPU

COMPUTE_TYPE: str = "float16" # float16 for CUDA, int8 for CPU


**Model Comparison:**

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| tiny | ⚡⚡⚡⚡⚡ | ⭐⭐ | Testing only |
| base | ⚡⚡⚡⚡ | ⭐⭐⭐ | Quick transcription |
| small | ⚡⚡⚡ | ⭐⭐⭐⭐ | Balanced |
| medium | ⚡⚡ | ⭐⭐⭐⭐⭐ | High accuracy |
| large-v2 | ⚡ | ⭐⭐⭐⭐⭐⭐ | **Medical use (recommended)** |

### VAD Settings

ENABLED: bool = True # Enable/disable Voice Activity Detection

THRESHOLD: float = 0.3 # Speech detection sensitivity (0.0-1.0)

**Threshold Guide:**
- `0.1-0.2`: Very sensitive (picks up whispers)
- `0.3`: **Recommended** (balanced)
- `0.4-0.5`: Less sensitive (filters background noise)

### Gemini AI Settings

MODEL_NAME: str = "gemini-2.5-flash"

TEMPERATURE: float = 0.2 # Response creativity (0.0-1.0)


**Temperature Guide:**
- `0.0-0.3`: **Medical use** (precise, consistent)
- `0.4-0.7`: Balanced
- `0.8-1.0`: Creative responses

### Server Settings

HOST: str = "0.0.0.0" # Listen on all network interfaces

PORT: int = 8000 # Server port

LOG_LEVEL: str = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL

---

## ▶️ Running the Server

### Start the Application

#### Ensure virtual environment is activated:

.venv\Scripts\Activate.ps1

#### Run the server:

python main.py


The server will start on `http://localhost:8000`

### Verify Server Status

Open your browser and navigate to:
- API Documentation: `http://localhost:8000/docs`
- WebSocket endpoint: `ws://localhost:8000/ws/transcribe`

---

## 🛠️ Troubleshooting

### Common Issues

**CUDA not found**
- Verify CUDA installation with `nvcc --version`
- Ensure NVIDIA GPU drivers are up to date
- Try reinstalling PyTorch with CUDA support

**MySQL connection failed**
- Verify MySQL is running
- Check credentials in `.env` file
- Ensure `medical_db` database exists

**Virtual environment issues**
- Delete `.venv` folder and recreate it
- Reinstall requirements from scratch

---

## 📚 Built With

### Core Technologies
- **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)** - Optimized Whisper implementation by SYSTRAN
- **[OpenAI Whisper](https://github.com/openai/whisper)** - Speech recognition foundation
- **[Silero VAD](https://github.com/snakers4/silero-vad)** - Voice Activity Detection
- **[Google Gemini](https://ai.google.dev/)** - AI-powered responses
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern web framework

### Infrastructure
- **PyTorch** - Deep learning framework
- **MySQL** - Database management
- **CUDA** - GPU acceleration

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code in accordance with the terms of the MIT License.

---

## ⚠️ Disclaimer

This software is provided "as is" without warranty of any kind. The author is not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use or misuse of this software.

**Medical Use Notice**: This tool is designed to assist healthcare professionals but should not replace professional medical judgment. Always verify AI-generated content before use in clinical settings.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## 👨‍💻 Author

**David M.**

---

<div align="center">

**Made with ❤️ for Healthcare Professionals**

If you find this project helpful, please consider giving it a ⭐

</div>
