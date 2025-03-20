from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
import datetime
import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Database setup
DATABASE_URL = "sqlite:///./appointments.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class Appointment(Base):
    __tablename__ = "appointments"
    
    id = Column(Integer, primary_key=True, index=True)
    service = Column(String)
    date = Column(String)
    time = Column(String)
    customer_name = Column(String)
    customer_phone = Column(String)
    notes = Column(Text, nullable=True)
    status = Column(String, default="pending")  # pending, confirmed, cancelled
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Schemas
class AppointmentBase(BaseModel):
    service: str
    date: str  # YYYY-MM-DD
    time: str  # HH:MM
    customer_name: str
    customer_phone: str
    notes: Optional[str] = None

class AppointmentCreate(AppointmentBase):
    pass

class AppointmentUpdate(BaseModel):
    service: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None

class AppointmentResponse(AppointmentBase):
    id: int
    status: str
    created_at: datetime.datetime
    
    class Config:
        orm_mode = True

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# FastAPI app
app = FastAPI(title="LLM Appointment Booking API")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# LLM Service
class LLMService:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("WARNING: GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
        self.client = genai.Client()
    
    def process_chat(self, message: str, db: Session):
        # Extract appointment intent from user message
        intent_data = self.extract_intent(message)
        
        # Generate appropriate response based on intent
        response = self.generate_response(message, intent_data, db)
        
        return response, intent_data
    
    def extract_intent(self, message: str):
        try:
            system_prompt = """
            Extract appointment booking details from the user message.
            Return a JSON with these fields:
            - intent: "booking", "reschedule", "cancel", "query", or "other"
            - service: type of service requested
            - date: date in YYYY-MM-DD format
            - time: time in HH:MM format
            - customer_name: the customer's name
            - customer_phone: the customer's phone number
            - notes: any special requests
            
            Only include fields that are explicitly mentioned in the message.
            """
            
            contents = [
                types.Content(
                    role="system",
                    parts=[types.Part.from_text(text=system_prompt)],
                ),
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message)],
                ),
            ]
            
            response = self.client.generate_content(
                model=self.model_name,
                contents=contents,
                generation_config=types.GenerationConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                )
            )
            
            # Parse the response
            try:
                if hasattr(response, 'text'):
                    return json.loads(response.text)
                else:
                    # Fallback for different response format
                    parts = response.candidates[0].content.parts
                    text_response = ''.join(part.text for part in parts)
                    return json.loads(text_response)
            except:
                return {"intent": "other"}
                
        except Exception as e:
            print(f"Error in LLM intent extraction: {str(e)}")
            return {"intent": "other"}
    
    def generate_response(self, user_message: str, intent_data: dict, db: Session):
        # Get available slots to use in response generation
        available_slots = []
        if intent_data.get("date"):
            # In a real app, query the DB for actual available slots
            available_slots = ["10:00", "11:30", "13:00", "15:30"]
        
        # Create system prompt with business context
        system_prompt = f"""
        You are an appointment booking assistant.
        
        User intent: {intent_data.get('intent', 'unknown')}
        Available slots for {intent_data.get('date', 'today')}: {available_slots}
        
        Respond conversationally to the user and help them book an appointment.
        If they want to book and a time is available, suggest it.
        If they want to book and their requested time is not available, suggest alternatives.
        Keep responses friendly but concise.
        """
        
        contents = [
            types.Content(
                role="system",
                parts=[types.Part.from_text(text=system_prompt)],
            ),
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)],
            ),
        ]
        
        try:
            response = self.client.generate_content(
                model=self.model_name,
                contents=contents,
                generation_config=types.GenerationConfig(
                    temperature=0.7,
                )
            )
            
            return response.text
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm having trouble processing your request. Please try again."

# Initialize LLM service
llm_service = LLMService()

# API Routes
@app.get("/")
async def root():
    return {"message": "LLM Appointment Booking API"}

# Chat endpoint to process natural language bookings
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    response, intent_data = llm_service.process_chat(request.message, db)
    
    # If it's a booking intent with sufficient information, create appointment automatically
    if (intent_data.get("intent") == "booking" and 
        intent_data.get("service") and 
        intent_data.get("date") and 
        intent_data.get("time") and 
        intent_data.get("customer_name")):
        
        # Create appointment from extracted data
        try:
            new_appointment = Appointment(
                service=intent_data["service"],
                date=intent_data["date"],
                time=intent_data["time"],
                customer_name=intent_data["customer_name"],
                customer_phone=intent_data.get("customer_phone", ""),
                notes=intent_data.get("notes", ""),
                status="pending"
            )
            db.add(new_appointment)
            db.commit()
        except Exception as e:
            print(f"Error creating appointment: {str(e)}")
    
    return {"response": response}

# CRUD API endpoints for admin interface
@app.get("/appointments/", response_model=List[AppointmentResponse])
async def read_appointments(
    date: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Appointment)
    if date:
        query = query.filter(Appointment.date == date)
    if status:
        query = query.filter(Appointment.status == status)
    return query.all()

@app.get("/appointments/{appointment_id}", response_model=AppointmentResponse)
async def read_appointment(appointment_id: int, db: Session = Depends(get_db)):
    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if appointment is None:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return appointment

@app.post("/appointments/", response_model=AppointmentResponse)
async def create_appointment(appointment: AppointmentCreate, db: Session = Depends(get_db)):
    db_appointment = Appointment(**appointment.dict())
    db.add(db_appointment)
    db.commit()
    db.refresh(db_appointment)
    return db_appointment

@app.put("/appointments/{appointment_id}", response_model=AppointmentResponse)
async def update_appointment(
    appointment_id: int, 
    appointment: AppointmentUpdate, 
    db: Session = Depends(get_db)
):
    db_appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if db_appointment is None:
        raise HTTPException(status_code=404, detail="Appointment not found")
    
    # Update appointment fields
    update_data = appointment.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_appointment, key, value)
    
    db.commit()
    db.refresh(db_appointment)
    return db_appointment

@app.delete("/appointments/{appointment_id}", response_model=AppointmentResponse)
async def delete_appointment(appointment_id: int, db: Session = Depends(get_db)):
    db_appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if db_appointment is None:
        raise HTTPException(status_code=404, detail="Appointment not found")
    
    db.delete(db_appointment)
    db.commit()
    return db_appointment

# Run with: uvicorn main:app --reload
