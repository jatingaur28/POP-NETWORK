# ⚖️ LegalEase Assistant

> AI-powered Legal Document Analysis & Assistance Platform

LegalEase Assistant is an intelligent legal-tech application designed to simplify complex legal documents using Generative AI and NLP. The platform enables users to upload legal documents, extract important information, generate simplified summaries, answer document-specific questions, and create calendar reminders for important legal dates.

---

## 🚀 Features

- 📄 Upload legal documents (PDF)
- 🤖 AI-powered document summarization
- 💬 Chat with your legal document
- 📅 Automatically extract important dates and generate calendar events
- 🔍 Clause and key information extraction
- ⚡ Fast and user-friendly Streamlit interface
- 🛡️ Secure document processing
- 📥 Download generated calendar (.ics) files

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### AI & NLP
- OpenAI GPT
- LangChain
- PyMuPDF
- DateParser
- Regular Expressions

### Calendar Integration
- ICS (iCalendar)

### Other Libraries
- Pandas
- Python-dotenv
- OS
- JSON

---

## 📂 Project Structure

```
Legal-Ease-Assistant/
│
├── app.py
├── requirements.txt
├── .env
├── assets/
├── uploads/
├── generated_files/
├── utils/
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/jatingaurx/Legal-Ease-Assistant.git

cd Legal-Ease-Assistant
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file.

```env
OPENAI_API_KEY=your_api_key_here
```

### Run the Project

```bash
streamlit run app.py
```

---

## 💡 How It Works

1. Upload a legal PDF.
2. AI extracts the document text.
3. The document is analyzed using LLMs.
4. Important clauses are summarized.
5. Key dates are detected automatically.
6. Users can ask questions related to the uploaded document.
7. Calendar events (.ics) can be downloaded for reminders.

---

## 🎯 Use Cases

- Contract Analysis
- Rental Agreements
- Employment Contracts
- NDA Review
- Legal Research
- Court Notice Understanding
- Policy Simplification

---

## 📸 Screenshots

Add screenshots here.

```
screenshots/
├── home.png
├── upload.png
├── summary.png
├── chatbot.png
```

---

## 🔮 Future Enhancements

- Multi-language support
- OCR for scanned PDFs
- Voice interaction
- Lawyer recommendation system
- Risk scoring for contracts
- RAG-based legal knowledge base
- Email notifications for legal deadlines

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to your branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Authors

**Jatin Gaur**

GitHub: https://github.com/jatingaurx

---

## ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.
