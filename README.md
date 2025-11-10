# SHL Assessment Recommendation System

This project is a SHL Assessment Recommendation System designed to help recruiters and hiring managers quickly find relevant SHL assessments based on natural language queries or job descriptions. It consists of two main components: a FastAPI backend and a React (Vite) frontend.

---

## 🚀 Live Links
- **Frontend App:** [https://shl-frontend-mu.vercel.app/](https://shl-frontend-mu.vercel.app/)
- **Backend API:** [https://shl-assessment-recommendation-system-1-4b8u.onrender.com](https://shl-assessment-recommendation-system-1-4b8u.onrender.com)
- **Frontend Repository:** [https://github.com/chinmoymaji08/shl_frontend](https://github.com/chinmoymaji08/shl_frontend)
- **Backend Repository:** [https://github.com/chinmoymaji08/SHL-Assessment-Recommendation-System](https://github.com/chinmoymaji08/SHL-Assessment-Recommendation-System)

---

## 🧩 Overview
The system accepts a job description or query, performs semantic similarity search using transformer-based embeddings, and returns the top SHL assessments with relevance scores and links to the SHL catalog.

---

## 🔑 Key Features
- Input: Free-text job descriptions or hiring needs.
- Output: 5–10 most relevant SHL assessments with scores.
- Built using **FastAPI** (backend) and **React + Vite + Tailwind CSS** (frontend).
- Uses **SentenceTransformers** for semantic similarity.
- Fully deployed and connected (Render + Vercel).

---

## 🛠️ Technology Stack
**Backend:** FastAPI, Python, SentenceTransformers, NumPy, Pandas  
**Frontend:** React (Vite), Tailwind CSS, Lucide React  
**Model:** `all-MiniLM-L6-v2` (sentence embeddings)  
**Deployment:** Render (backend), Vercel (frontend)  

---

## 🌐 API Endpoints
| Method | Endpoint | Description |
|---------|-----------|-------------|
| **GET** | `/health` | Returns API health and model status |
| **POST** | `/recommend` | Accepts a job query and returns top SHL assessments |

Example request:
```bash
POST /recommend
{
  "query": "Hiring a Python developer with analytical and communication skills"
}
```

Example response:
```json
{
  "query": "Python developer assessment",
  "recommendations": [
    {
      "assessment_name": "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
      "relevance_score": 0.98
    }
  ],
  "timestamp": "2025-11-10T01:02:32.483286",
  "processing_time_ms": 3855.2
}
```

---

## ⚙️ Local Setup

### 1. Clone Repositories
```bash
git clone https://github.com/chinmoymaji08/SHL-Assessment-Recommendation-System.git
git clone https://github.com/chinmoymaji08/shl_frontend.git
```

### 2. Backend Setup
```bash
cd SHL-Assessment-Recommendation-System
pip install -r requirements.txt
python main.py
```
Access docs at → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Frontend Setup
```bash
cd ../shl_frontend
npm install
npm run dev
```
Frontend runs at → [http://localhost:5173](http://localhost:5173)

---

## ☁️ Deployment

### **Frontend (Vercel)**
1. Connect the `shl_frontend` GitHub repo.
2. Add environment variable:
   ```
   VITE_API_BASE = https://shl-assessment-recommendation-system-1-4b8u.onrender.com
   ```
3. Build settings:
   - **Build Command:** `vite build`
   - **Output Directory:** `dist`

### **Backend (Render)**
1. Connect the backend repo.
2. Use `pip install -r requirements.txt` as the build command.
3. Set the start command:
   ```
   uvicorn main:app --host 0.0.0.0 --port 10000
   ```
4. Add environment variables as needed.

---

## 📊 Example Use Case
| Input Query | Example Output |
|--------------|----------------|
| “Hiring data analysts with SQL and reasoning skills” | Suggests SHL Cognitive Ability Test, Global Skills Assessment, Data Analysis Assessment |
| “Need an assessment for software engineers proficient in Python” | Suggests SHL Python Assessment, Global Skills Assessment, Personality Questionnaire |

---

## 👤 Author
**Chinmoy Maji**  
📍 Tamluk, India  
🎓 B.Tech in Computer Science & Engineering (AI & ML)  
💼 Passionate about Machine Learning & AI Engineering  
🔗 [LinkedIn](https://www.linkedin.com/in/chinmoymaji/)
