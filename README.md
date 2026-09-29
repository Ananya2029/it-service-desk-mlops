# Enterprise IT Service Desk — Automated Ticket Routing API

Routes an incoming IT support ticket to the right resolver team automatically, from its subject and description, and serves the model as a REST API ready to deploy on Render.

## Problem

In a large service desk, tickets are often assigned to the wrong team first and bounce between queues, which delays resolution. Predicting the correct **assignment group** at creation time removes that manual triage step.

## Model

| | |
|---|---|
| Input | Ticket `subject` + `description` (free text) |
| Output | One of **10 assignment groups** (Access & Identity, Application, Cloud & Infrastructure, Cybersecurity, Database, Email & Collaboration, Enterprise Systems, Hardware, Network, Software) + confidence |
| Pipeline | Text cleaning → TF-IDF (unigrams + bigrams, 50k features, sublinear TF) → Logistic Regression |
| Training data | IT service desk ticket dataset, stratified 80/20 train/test split |

## Results

| Version | Test data | Accuracy |
|---|---|---|
| V1 | Standard tickets | 100% — the classes were too cleanly separated to be a realistic test |
| **V2 (deployed)** | Retrained with deliberately **ambiguous, overlapping tickets** (4,000 held-out) | **87.2%** |

V1's perfect score was a warning sign rather than a success: real tickets are messy and mention several systems at once. V2 adds challenging examples so the reported accuracy reflects realistic routing difficulty.

Example predictions from the deployed model:

| Ticket | Routed to | Confidence |
|---|---|---|
| "VPN not connecting — error 809 from home" | Network Support | 98.8% |
| "Password reset — locked out after too many attempts" | Access & Identity Management | 94.6% |
| "Outlook not syncing — emails not arriving" | Email & Collaboration Support | 95.4% |
| "Laptop screen flickering and went black" | Hardware Support | 97.6% |

## API

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Interactive docs at http://127.0.0.1:8000/docs.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check + model version |
| GET | `/model-info` | Model metadata (features, hyperparameters, accuracy, classes) |
| POST | `/predict` | Route a ticket |

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"subject": "VPN not connecting", "description": "Cannot connect to company VPN from home, error 809"}'
```

```json
{"predicted_assignment_group": "Network Support", "confidence": 0.988, "model_version": "V2"}
```

## Deployment

`render.yaml` deploys the API to [Render](https://render.com) as a web service (`uvicorn app:app --host 0.0.0.0 --port $PORT`). `scikit-learn` is pinned to the version the model was saved with, so the pickled pipeline loads identically in production.

## Files

| File | Description |
|---|---|
| `app.py` | FastAPI application |
| `it_service_desk_ticket_router.pkl` | Trained TF-IDF + Logistic Regression pipeline |
| `model_info.json` | Model metadata and evaluation |
| `render.yaml` | Render deployment config |

Training notebook: [`Enterprise_IT_Service_Desk_Intelligence.ipynb`](https://github.com/Ananya2029/Deep_Learning_Project/blob/main/Enterprise_IT_Service_Desk_Intelligence.ipynb)

## Tech Stack

Python · Scikit-learn · TF-IDF · Logistic Regression · FastAPI · Pydantic · Uvicorn · Render
