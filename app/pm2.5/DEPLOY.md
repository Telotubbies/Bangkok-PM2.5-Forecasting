# Deployment Guide — BKK PM2.5 Air Quality App

## Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌───────────────────┐
│  Namecheap   │────▶│   Cloudflare    │────▶│     Vercel        │
│  (Domain)    │     │  (DNS + CDN)    │     │  (React Frontend) │
│  pm25bkk.com │     │  Proxy + Cache  │     │  Vite + React 18  │
└──────────────┘     └─────────────────┘     └────────┬──────────┘
                                                      │
                           ┌──────────────────────────┼──────────────────┐
                           │                          │                  │
                    ┌──────▼───────┐          ┌───────▼──────┐   ┌──────▼────────┐
                    │   Supabase   │          │ Hugging Face │   │  Cloudflare   │
                    │  (Database)  │          │  (ST-UNN AI) │   │    (R2/KV)    │
                    │  PostgreSQL  │          │  Inference   │   │  Static cache │
                    │  + RLS       │          │  Endpoint    │   │               │
                    └──────────────┘          └──────────────┘   └───────────────┘
```

---

## Step 1: Supabase (Database)

1. Go to [supabase.com](https://supabase.com) → Create new project
2. Choose region: **Singapore** (closest to Bangkok)
3. Save the **Project URL** and **anon key** from Settings → API
4. Open SQL Editor and run: `supabase/schema.sql`
5. Verify tables created: `stations`, `readings`, `forecasts`, `hotspots`

### Seed initial station data

```sql
INSERT INTO stations (id, name, name_th, lat, lon, area) VALUES
  ('s01', 'Din Daeng',   'ดินแดง',   13.7649, 100.5440, 'Din Daeng'),
  ('s02', 'Ratchathewi', 'ราชเทวี',  13.7583, 100.5316, 'Ratchathewi'),
  ('s03', 'Bang Na',     'บางนา',    13.6673, 100.6048, 'Bang Na'),
  ('s04', 'Lat Phrao',   'ลาดพร้าว', 13.8034, 100.5702, 'Lat Phrao'),
  ('s05', 'Thon Buri',   'ธนบุรี',   13.7220, 100.4872, 'Thon Buri'),
  ('s06', 'Chatuchak',   'จตุจักร',  13.8200, 100.5536, 'Chatuchak'),
  ('s07', 'Bang Kapi',   'บางกะปิ',  13.7658, 100.6456, 'Bang Kapi'),
  ('s08', 'Khlong Toei', 'คลองเตย',  13.7130, 100.5578, 'Khlong Toei'),
  ('s09', 'Prawet',      'ประเวศ',   13.6870, 100.6890, 'Prawet'),
  ('s10', 'Nong Khaem',  'หนองแขม',  13.7040, 100.3530, 'Nong Khaem');
```

---

## Step 2: Hugging Face (AI Model)

### Option A: Inference Endpoint (recommended for production)

1. Go to [huggingface.co/new-inference-endpoint](https://huggingface.co/new-inference-endpoint)
2. Upload the `stunn_deployment.pt` model bundle from `model_training.ipynb`
3. Create a custom handler:

```python
# handler.py for HF Inference Endpoint
import torch
import json

class EndpointHandler:
    def __init__(self, path):
        bundle = torch.load(f"{path}/stunn_deployment.pt", map_location="cpu")
        self.model = bundle["model"]
        self.model.eval()
        self.norm_stats = bundle["normalization_stats"]
        self.feature_cols = bundle["feature_columns"]

    def __call__(self, data):
        inputs = data.get("inputs", {})
        features = torch.tensor(inputs["features"], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(features)
        return {
            "predictions": [
                {"horizon_hours": h, "pm25": round(v, 1), "confidence": 0.88}
                for h, v in zip(inputs.get("horizons", [24, 72]), pred[0].tolist())
            ],
            "model_version": "stunn-v1"
        }
```

4. Set region to **eu-west-1** or **us-east-1**
5. Copy the endpoint URL to `VITE_HF_INFERENCE_URL`

### Option B: Hugging Face Spaces (free tier)

1. Create a Space with Gradio or FastAPI
2. Deploy the model as a simple API
3. Use the Space URL as `VITE_HF_INFERENCE_URL`

---

## Step 3: Vercel (Frontend Hosting)

1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```

2. From the `app/pm2.5` directory:
   ```bash
   vercel login
   vercel
   ```

3. Set environment variables in Vercel dashboard (Settings → Environment Variables):
   ```
   VITE_SUPABASE_URL       = https://your-project.supabase.co
   VITE_SUPABASE_ANON_KEY  = eyJhbGciOiJI...
   VITE_HF_INFERENCE_URL   = https://your-endpoint.huggingface.cloud
   VITE_HF_API_TOKEN       = hf_...
   ```

4. Deploy:
   ```bash
   vercel --prod
   ```

The `vercel.json` config handles SPA routing and caching automatically.

---

## Step 4: Namecheap (Domain)

1. Buy a domain at [namecheap.com](https://www.namecheap.com) (e.g., `pm25bkk.com`)
2. **Do NOT** use Namecheap DNS — we'll use Cloudflare instead
3. Under Domain → Nameservers, select **Custom DNS**
4. Enter the Cloudflare nameservers (see Step 5)

---

## Step 5: Cloudflare (DNS + CDN + Security)

1. Go to [cloudflare.com](https://cloudflare.com) → Add site → enter your domain
2. Select **Free plan**
3. Cloudflare will give you 2 nameservers — enter these in Namecheap (Step 4)
4. Add DNS records:

   | Type  | Name | Content                        | Proxy |
   |-------|------|--------------------------------|-------|
   | CNAME | @    | cname.vercel-dns.com           | ✅    |
   | CNAME | www  | cname.vercel-dns.com           | ✅    |

5. In Vercel dashboard → Settings → Domains → Add your domain
6. Vercel will verify the DNS and issue SSL automatically

### Cloudflare Recommended Settings

- **SSL/TLS**: Full (strict)
- **Always Use HTTPS**: On
- **Auto Minify**: HTML, CSS, JS
- **Brotli**: On
- **Cache Level**: Standard
- **Browser Cache TTL**: 4 hours

### Cloudflare Page Rules (optional)

```
*pm25bkk.com/assets/*  → Cache Level: Cache Everything, Edge TTL: 1 month
*pm25bkk.com/api/*     → Cache Level: Bypass
```

---

## Environment Variables Summary

| Variable                 | Where           | Description                         |
|--------------------------|-----------------|-------------------------------------|
| `VITE_SUPABASE_URL`     | Vercel + local  | Supabase project URL                |
| `VITE_SUPABASE_ANON_KEY`| Vercel + local  | Supabase anonymous API key          |
| `VITE_HF_INFERENCE_URL` | Vercel + local  | Hugging Face model endpoint URL     |
| `VITE_HF_API_TOKEN`     | Vercel + local  | Hugging Face API token              |

For local development, copy `.env.example` to `.env`:

```bash
cp .env.example .env
# Edit .env with your actual values
```

---

## Data Pipeline (keeping data fresh)

The Supabase database needs regular data ingestion. Options:

1. **Supabase Edge Functions** — cron job to fetch Air4Thai API every hour
2. **GitHub Actions** — scheduled workflow running the ingestion notebook
3. **Hugging Face Spaces** — background worker for FIRMS hotspot data

### Suggested cron schedule

| Task                    | Frequency    | Source            |
|-------------------------|-------------|-------------------|
| PM2.5 station readings  | Every hour  | Air4Thai API      |
| Weather data            | Every 6 hrs | Open-Meteo API    |
| Hotspot detection       | Every 6 hrs | NASA FIRMS API    |
| Model re-forecast       | Every 6 hrs | HF Inference      |

---

## Monitoring

- **Vercel**: Analytics dashboard (built-in)
- **Supabase**: Database metrics in dashboard
- **Cloudflare**: Web Analytics + security events
- **Hugging Face**: Inference endpoint metrics
