
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

with open("it_service_desk_ticket_router.pkl", "rb") as file:
    model = pickle.load(file)

with open("model_info.json", "r") as file:
    model_info = json.load(file)

app = FastAPI(
    title="IT Service Desk Ticket Router API",
    description="API for automated enterprise IT service desk ticket routing",
    version="1.0"
)

class TicketRequest(BaseModel):
    subject: str
    description: str

@app.get("/")
def home():
    return {
        "message": "IT Service Desk Ticket Router API is running",
        "model_version": model_info["model_version"]
    }

@app.get("/model-info")
def get_model_info():
    return model_info

@app.post("/predict")
def predict_ticket(ticket: TicketRequest):

    ticket_text = ticket.subject + " " + ticket.description

    prediction = model.predict([ticket_text])[0]

    probabilities = model.predict_proba([ticket_text])[0]

    confidence = float(max(probabilities))

    return {
        "predicted_assignment_group": prediction,
        "confidence": round(confidence, 4),
        "model_version": model_info["model_version"]
    }
