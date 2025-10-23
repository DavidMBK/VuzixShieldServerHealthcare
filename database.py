# database.py

from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# MySQL configuration
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = os.getenv("MYSQL_PORT")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# Connection string for MySQL
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connection before using it
    pool_recycle=3600,   # Reconnect every hour
    echo=True            # SQL query logging (disable in production)
)

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), index=True)
    age = Column(Integer)
    blood_type = Column(String(10))
    allergies = Column(Text)

class BloodTest(Base):
    __tablename__ = "blood_tests"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_name = Column(String(100), index=True)
    test_date = Column(Date)
    hemoglobin = Column(Float)  # g/dL
    white_blood_cells = Column(Float)  # 10^3/μL
    platelets = Column(Float)  # 10^3/μL
    glucose = Column(Float)  # mg/dL
    cholesterol = Column(Float)  # mg/dL
    notes = Column(Text)

class Prescription(Base):
    __tablename__ = "prescriptions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_name = Column(String(100), index=True)
    medication = Column(String(200))
    dosage = Column(String(50))
    frequency = Column(String(100))
    start_date = Column(Date)
    notes = Column(Text)

class MedicalHistory(Base):
    __tablename__ = "medical_history"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_name = Column(String(100), index=True)
    condition = Column(String(200))
    diagnosed_date = Column(Date)
    status = Column(String(50))  # Active, Resolved, Chronic
    notes = Column(Text)

def init_db():
    """Creates tables and populates with sample data for Mario, Luigi, and Peach."""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("MySQL tables created successfully!")
        
        db = SessionLocal()
        
        # Check if DB is already populated
        if db.query(Patient).count() > 0:
            print("ℹ️ Database already populated.")
            db.close()
            return
        
        print("🏥 Populating database with patients: Mario, Luigi, Peach...")
        
        # ========== PATIENT 1: MARIO ==========
        mario = Patient(
            name="Mario",
            age=35,
            blood_type="O+",
            allergies="No known allergies"
        )
        
        mario_blood = BloodTest(
            patient_name="Mario",
            test_date=datetime.date(2025, 10, 5),
            hemoglobin=15.2,
            white_blood_cells=6.8,
            platelets=240,
            glucose=92,
            cholesterol=185,
            notes="Optimal values, patient in good health"
        )
        
        mario_prescription = Prescription(
            patient_name="Mario",
            medication="Vitamin D3",
            dosage="1000 IU",
            frequency="Once daily",
            start_date=datetime.date(2025, 9, 20),
            notes="Preventive supplementation"
        )
        
        mario_history = MedicalHistory(
            patient_name="Mario",
            condition="Right wrist fracture",
            diagnosed_date=datetime.date(2023, 8, 15),
            status="Resolved",
            notes="Complete recovery after 6 weeks in cast"
        )
        
        # ========== PATIENT 2: LUIGI ==========
        luigi = Patient(
            name="Luigi",
            age=32,
            blood_type="A-",
            allergies="Penicillin, Lactose"
        )
        
        luigi_blood_1 = BloodTest(
            patient_name="Luigi",
            test_date=datetime.date(2025, 10, 12),
            hemoglobin=13.5,
            white_blood_cells=9.2,
            platelets=210,
            glucose=105,
            cholesterol=245,
            notes="Elevated cholesterol, diet recommended"
        )
        
        luigi_blood_2 = BloodTest(
            patient_name="Luigi",
            test_date=datetime.date(2025, 9, 10),
            hemoglobin=13.8,
            white_blood_cells=8.5,
            platelets=220,
            glucose=98,
            cholesterol=250,
            notes="Cholesterol values to be monitored"
        )
        
        luigi_prescription_1 = Prescription(
            patient_name="Luigi",
            medication="Atorvastatin",
            dosage="20mg",
            frequency="Once daily (evening)",
            start_date=datetime.date(2025, 9, 15),
            notes="For high cholesterol control"
        )
        
        luigi_prescription_2 = Prescription(
            patient_name="Luigi",
            medication="Cetirizine",
            dosage="10mg",
            frequency="Once daily",
            start_date=datetime.date(2025, 10, 1),
            notes="Antihistamine for seasonal allergies"
        )
        
        luigi_history = MedicalHistory(
            patient_name="Luigi",
            condition="Hypercholesterolemia",
            diagnosed_date=datetime.date(2025, 8, 20),
            status="Active",
            notes="Under pharmacological treatment and controlled diet"
        )
        
        # ========== PATIENT 3: PEACH ==========
        peach = Patient(
            name="Peach",
            age=28,
            blood_type="B+",
            allergies="Pollen, Peanuts"
        )
        
        peach_blood = BloodTest(
            patient_name="Peach",
            test_date=datetime.date(2025, 10, 14),
            hemoglobin=12.8,
            white_blood_cells=7.1,
            platelets=265,
            glucose=88,
            cholesterol=175,
            notes="All values within normal range"
        )
        
        peach_prescription = Prescription(
            patient_name="Peach",
            medication="Ibuprofen",
            dosage="600mg",
            frequency="3 times daily after meals",
            start_date=datetime.date(2025, 10, 15),
            notes="For lower back pain due to muscle tension"
        )
        
        peach_history_1 = MedicalHistory(
            patient_name="Peach",
            condition="Chronic low back pain",
            diagnosed_date=datetime.date(2024, 3, 10),
            status="Chronic",
            notes="Managed with physical therapy and occasional NSAIDs"
        )
        
        peach_history_2 = MedicalHistory(
            patient_name="Peach",
            condition="Allergic rhinitis",
            diagnosed_date=datetime.date(2022, 5, 5),
            status="Chronic",
            notes="Seasonal flare-ups in spring"
        )
        
        # Add all records to database
        db.add_all([
            mario, mario_blood, mario_prescription, mario_history,
            luigi, luigi_blood_1, luigi_blood_2, luigi_prescription_1, 
            luigi_prescription_2, luigi_history,
            peach, peach_blood, peach_prescription, peach_history_1, peach_history_2
        ])
        
        db.commit()
        print("Database successfully populated: Mario, Luigi, Peach!")
        db.close()
        
    except Exception as e:
        print(f"Error during database initialization: {e}")
        raise

def get_db():
    """Dependency to obtain a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
