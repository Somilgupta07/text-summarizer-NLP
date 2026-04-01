# 📝 Text Summarizer using T5 | FastAPI + HTML

A complete **end-to-end NLP project** that performs **abstractive text summarization** using a fine-tuned **T5 Transformer model**.  
This project is built with **PyTorch**, **Hugging Face Transformers**, and **FastAPI**, and includes a simple **HTML frontend** for user interaction.

It takes long dialogues or text as input and generates concise, meaningful summaries.

---

## 🚀 Project Overview

This project demonstrates how to build a **real-world NLP application** by:

- Preparing and preprocessing text data
- Fine-tuning a **T5 model** for summarization
- Saving the trained model locally
- Building a **FastAPI backend**
- Connecting it to an **HTML frontend**
- Serving summaries through both **web UI** and **API endpoint**

This is a complete **machine learning deployment workflow** project.

---

## ✨ Features

- 🔹 Abstractive text summarization using **T5**
- 🔹 Fine-tuned transformer-based model
- 🔹 Clean text preprocessing pipeline
- 🔹 Simple and user-friendly **web interface**
- 🔹 REST API endpoint for summarization
- 🔹 FastAPI auto-generated API docs
- 🔹 Easy to run locally
- 🔹 Good beginner-to-intermediate **NLP deployment project**

---

## 🛠️ Tech Stack

### Programming Language

- **Python**

### Machine Learning / NLP

- **PyTorch**
- **Hugging Face Transformers**
- **T5 (Text-To-Text Transfer Transformer)**

### Backend

- **FastAPI**
- **Uvicorn**

### Frontend

- **HTML**
- **Jinja2 Templates**
- **CSS (optional styling)**

---

## 📂 Project Structure

```bash
Text-Summarizer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html
│
├── saved_summary_model/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors          # or pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── spiece.model
│   └── ...
│
├── notebooks/
│   └── text_summarizer.ipynb
│
└── screenshots/
    ├── home_page.png
    └── summary_output.png


🔍 Learning Outcomes / What This Project Demonstrates

This project demonstrates practical knowledge of:

NLP model fine-tuning
Transformer-based summarization
T5 architecture usage
Tokenization and sequence generation
PyTorch model loading and inference
FastAPI backend development
HTML template integration with backend
End-to-end ML application deployment

This makes it a strong project for:

AI/ML internships
college projects
resume portfolio
GitHub showcase
📈 Future Improvements

This project can be improved further by adding:

🎨 Better frontend UI using Bootstrap / Tailwind CSS
☁️ Deployment on Render / Railway / Hugging Face Spaces
📄 File upload support (.txt, .pdf)
🎛️ Summary length control (short / medium / detailed)
🔁 Compare multiple summarization models
🌍 Support for multilingual summarization
📊 ROUGE score evaluation on test set
🧪 User input validation and error handling
🗂️ Logging and production-ready structure
```
