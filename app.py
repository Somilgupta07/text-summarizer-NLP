from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

# Templates folder
templates = Jinja2Templates(directory="templates")

# Pydantic model for API
class DialogueInput(BaseModel):
    dialogue: str

# Clean text
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

# Summarization function
def summarize_dialogue(dialogue: str) -> str:
    dialogue = "summarize: " + clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    # move tensors to device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

# ---------------- API endpoint ----------------
@app.post("/summarize/")
async def summarize_api(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

# ---------------- UI GET route ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"summary": ""})

# ---------------- UI POST route ----------------
@app.post("/", response_class=HTMLResponse)
async def summarize_form(request: Request, dialogue: str = Form(...)):
    summary = summarize_dialogue(dialogue)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "summary": summary,
            "dialogue": dialogue
        }
    )