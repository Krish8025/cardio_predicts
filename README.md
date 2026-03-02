# 🫀 Cardio Disease Prediction

A full-stack machine learning web application that predicts cardiovascular disease risk based on patient health metrics.

## 🚀 Live Demo

- **Frontend**: Deployed on [Vercel](https://vercel.com)
- **Backend API**: Deployed on [Render](https://render.com)

---

## 📁 Project Structure

```
ML_Project_cardio/
├── backend/          # FastAPI backend (Python ML model)
│   ├── main.py       # API server
│   ├── train_improved_model.py
│   ├── requirements.txt
│   └── Procfile      # For Render deployment
└── cardio/           # Next.js frontend (TypeScript)
    ├── app/
    ├── components/
    └── package.json
```

---

## 🧠 Model

- **Algorithm**: Gradient Boosting (Optimized)
- **Features**: Age, Gender, Height, Weight, Blood Pressure, Cholesterol, Glucose, Smoking, Alcohol, Activity
- **Feature Engineering**: BMI, BP Ratio, Interaction features (smoke×age, smoke×bp, etc.)

---

## ⚙️ Local Setup

### Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate       # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`  
Swagger docs at `http://localhost:8000/docs`

### Frontend

```bash
cd cardio
npm install
# Create .env.local and set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Frontend will be available at `http://localhost:3000`

---

## 🌐 Deployment

### Backend (Render)
1. Connect repo to [render.com](https://render.com)
2. Set **Root Directory** to `backend`
3. Set **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Set **Environment**: Python 3

### Frontend (Vercel)
1. Connect repo to [vercel.com](https://vercel.com)
2. Set **Root Directory** to `cardio`
3. Add environment variable: `NEXT_PUBLIC_API_URL=<your-render-backend-url>`

---

## 📊 API Reference

### `POST /predict`

**Request Body:**
```json
{
  "age": 18250,
  "gender": 2,
  "height": 170,
  "weight": 70.0,
  "ap_hi": 120,
  "ap_lo": 80,
  "cholesterol": 1,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.12,
  "status": "Low Risk"
}
```
