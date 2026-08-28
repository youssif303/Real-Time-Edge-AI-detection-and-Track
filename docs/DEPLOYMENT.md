# Deployment

EdgeTrack is split across two services:

- Vercel hosts the React dashboard from the `frontend/` root directory.
- Render hosts the FastAPI and YOLO service from the repository root.

## Render backend

Create a Render Web Service connected to this repository, or use `render.yaml`.

- Runtime: `Python 3`
- Build command: `pip install -r backend/requirements.txt`
- Start command: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Environment variable: `EDGETRACK_CORS_ORIGINS=https://YOUR-VERCEL-DOMAIN.vercel.app`

The first request may download the YOLO weights. Render's filesystem is ephemeral, so generated annotated videos are demo artifacts rather than permanent storage.

## Vercel frontend

Import the same repository as a Vercel project and set its Root Directory to `frontend`. The checked-in `frontend/vercel.json` supplies the Vite build and `dist` output settings.

Set this environment variable in Vercel:

```text
VITE_API_URL=https://YOUR-RENDER-SERVICE.onrender.com
```

Deploy the backend first, copy its public URL into both environment settings, then deploy the frontend. Verify `/health` and upload a short clip from the public dashboard.