# AROHA - India IDSP Disease Surveillance Chat App

A Next.js-based chat application for querying India's IDSP disease surveillance data.

## Prerequisites

- Node.js 18+ installed
- MongoDB running locally on port 27017
- **Flask backend (Aroha folder)** - Get this from the company separately
- Gemini API key
- Firebase project (for authentication)

## Backend Setup (Important!)

This frontend requires the Flask backend to be running. Get the `Aroha` folder from your team/company and:

1. Place it in a separate directory (e.g., `f:\Projects\Aroha`)
2. Install Flask dependencies: `pip install -r requirements.txt`
3. Set up the `.env` file with `GEMINI_API_KEY` and `PC_API_Key`
4. Run: `python backend.py`
5. Backend should be accessible at `http://localhost:2000`

## Setup Instructions

### 1. Install Dependencies
```bash
npm install
```

### 2. Environment Variables
Copy `.env.example` to `.env.local` and fill in your actual values:
```bash
cp .env.example .env.local
```

Required environment variables:
- `MONGODB_URI` - MongoDB connection string
- `GEMINI_API_KEY` - Your Google Gemini API key
- `JWT_SECRET` - Secret key for JWT tokens
- `NEXTAUTH_SECRET` - Secret for NextAuth
- Firebase configuration (API key, project ID, etc.)

### 3. Backend Server
Make sure the Flask backend is running on `http://localhost:2000`

The backend should have:
- `/Chatbot` endpoint that accepts POST requests
- Connected to IDSP database with disease surveillance data

### 4. Run Development Server
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the app.

### 5. Build for Production
```bash
npm run build
npm start
```

## Features

- Real-time chat interface for disease surveillance queries
- User authentication (email/password + Google Sign-In)
- Conversation history saved to MongoDB
- Export chat functionality
- Responsive design with dark theme

## Tech Stack

- **Frontend:** Next.js 14, React, Tailwind CSS
- **Backend:** Flask (separate repository)
- **Database:** MongoDB
- **Auth:** NextAuth.js, Firebase
- **AI:** Google Gemini API

## Important Notes

⚠️ **Never commit `.env.local` or any file containing API keys!**

⚠️ **Backend Connection:** This app requires the Flask backend to be running locally on port 2000. For production deployment, update the Flask endpoint in `app/api/chat/route.js`.

## License

[Your License Here]
