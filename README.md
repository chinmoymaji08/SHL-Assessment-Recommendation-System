# SHL Assessment Recommendation System

This project is a SHL Assessment Recommendation System designed to help recruiters and hiring managers quickly find relevant SHL assessments based on natural language queries or job descriptions. It consists of two main components: a FastAPI backend and a React (Vite) frontend.

---

## 🚀 Live Links
- **Frontend App:** [https://shl-frontend-mu.vercel.app/](https://shl-frontend-mu.vercel.app/)
- **Frontend Repository:** [https://github.com/chinmoymaji08/shl_frontend](https://github.com/chinmoymaji08/shl_frontend)

---

## 🧩 Overview
The system accepts a query or job description, analyzes it semantically, and returns 5–10 of the most relevant SHL assessments from the SHL product catalog, each with a relevance score and direct link.

---

## 🔑 Key Features
- Takes job descriptions or natural language queries as input.
- Recommends 5–10 relevant SHL assessments.
- Displays assessment name, URL, and relevance score.
- Backend built with FastAPI and semantic search using SentenceTransformer embeddings.
- Frontend developed using React (Vite) and Tailwind CSS, deployed on Vercel.

---

## 🛠️ Technology Stack
**Backend:** FastAPI, Python, SentenceTransformers, NumPy, Pandas  
**Frontend:** React, Vite, Tailwind CSS, Lucide React  
**Model:** all-MiniLM-L6-v2 for embeddings  
**Deployment:** Vercel (frontend), Render/Railway (backend possible)

---

## 🌐 Backend API Endpoints
1. **GET /health** → Returns API health status.  
2. **POST /recommend** → Takes a text query and returns top 10 relevant SHL assessments.

---

## ⚙️ How to Run Locally

### 1. Clone both repositories
```bash
git clone https://github.com/chinmoymaji08/shl_frontend.git
cd "SHL Assessment Recommendation System"
```

### 2. Backend setup
```bash
pip install -r requirements.txt
python main.py
```
Runs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Frontend setup
```bash
cd shl_frontend
npm install
npm run dev
```
Runs locally at: [http://localhost:5173](http://localhost:5173)

---

## ☁️ Deployment

The frontend is deployed on **Vercel** using Vite.  
To redeploy:
1. Connect your GitHub repo.  
2. Set:
   - **Build Command:** `vite build`
   - **Output Directory:** `dist`

The backend can be deployed on **Render** or **Railway** for free.

---

## 📊 Generating Predictions
To generate predictions on the test dataset:
```bash
python generate_predictions.py --test-file data/test_unlabeled.csv --output-file predictions.csv
```
Outputs a `predictions.csv` file with top recommendations per query.

---

## 👤 Author
**Chinmoy Maji**  
📍 Tamluk, India  
🎓 B.Tech in Computer Science and Engineering (AI & ML)  
💼 Passionate about Machine Learning Engineering  
🔗 [LinkedIn](https://linkedin.com/in/chinmoy-maji08)
