# Deploy Guide — ETH Signal Bot

Two parts:
1. **Backend API** → Railway (free, always on, Python)
2. **Frontend**    → Netlify (free, always on, static)

---

## Step 1 — Deploy the API to Railway

1. Go to https://railway.app and sign up (free)
2. Click **New Project → Deploy from GitHub repo**
3. Push your code to GitHub first (or use Railway's CLI)
4. Set the **root directory** to `api/`
5. Railway will auto-detect Python and use the `Procfile`
6. Once deployed, copy your public URL — it looks like:
   `https://eth-bot-api-production.up.railway.app`

**That's your API_URL.**

---

## Step 2 — Set the API URL in the frontend

Open `frontend/index.html` and find this line near the bottom:

```js
const API_URL = "https://YOUR-API-URL.railway.app/signal";
```

Replace it with your actual Railway URL:

```js
const API_URL = "https://eth-bot-api-production.up.railway.app/signal";
```

---

## Step 3 — Deploy the frontend to Netlify

1. Go to https://netlify.com and sign up (free)
2. Click **Add new site → Deploy manually**
3. Drag and drop your `frontend/` folder onto the page
4. Netlify gives you a URL like `https://eth-signal-bot.netlify.app`

Done. The site runs 24/7, refreshes the signal every 60 seconds.

---

## Folder structure

```
RICH/
├── api/
│   ├── main.py          ← FastAPI backend (deploy to Railway)
│   ├── requirements.txt
│   └── Procfile
├── frontend/
│   └── index.html       ← Static site (deploy to Netlify)
├── bot.py               ← Original terminal bot
└── requirements.txt
```

---

## Test the API locally

```bash
cd api
pip install fastapi uvicorn requests pandas numpy
uvicorn main:app --reload
```

Then open http://localhost:8000/signal in your browser.
