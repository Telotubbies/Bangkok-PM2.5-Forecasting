/**
 * Hugging Face Inference API client for the ST-UNN PM2.5 forecasting model.
 *
 * Architecture:
 *   Browser → Vercel Edge (CORS proxy) → Hugging Face Inference Endpoint
 *
 * The model accepts a JSON payload with 30 days of feature vectors (61 features)
 * and returns predicted PM2.5 values for +1d and +3d horizons.
 */

const HF_API_URL = import.meta.env.VITE_HF_INFERENCE_URL as string | undefined;
const HF_API_TOKEN = import.meta.env.VITE_HF_API_TOKEN as string | undefined;

export interface ForecastRequest {
  station_id: string;
  features: number[][];
  horizons: number[];
}

export interface ForecastResponse {
  station_id: string;
  predictions: {
    horizon_hours: number;
    pm25: number;
    confidence: number;
  }[];
  model_version: string;
  inference_ms: number;
}

export async function requestForecast(
  req: ForecastRequest,
): Promise<ForecastResponse | null> {
  if (!HF_API_URL || !HF_API_TOKEN) {
    console.warn("[forecast-api] HF credentials not set — returning mock forecast");
    return mockForecast(req);
  }

  try {
    const response = await fetch(HF_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${HF_API_TOKEN}`,
      },
      body: JSON.stringify({
        inputs: {
          station_id: req.station_id,
          features: req.features,
          horizons: req.horizons,
        },
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error("[forecast-api] HF error:", response.status, errText);
      return mockForecast(req);
    }

    const data = await response.json();
    return data as ForecastResponse;
  } catch (err) {
    console.error("[forecast-api] network error:", err);
    return mockForecast(req);
  }
}

/**
 * Batch forecast for multiple stations.
 */
export async function requestBatchForecast(
  requests: ForecastRequest[],
): Promise<Map<string, ForecastResponse>> {
  const results = new Map<string, ForecastResponse>();
  const promises = requests.map(async (req) => {
    const res = await requestForecast(req);
    if (res) results.set(req.station_id, res);
  });
  await Promise.all(promises);
  return results;
}

function mockForecast(req: ForecastRequest): ForecastResponse {
  return {
    station_id: req.station_id,
    predictions: req.horizons.map((h) => ({
      horizon_hours: h,
      pm25: Math.round(45 + Math.random() * 50),
      confidence: 0.82 + Math.random() * 0.12,
    })),
    model_version: "stunn-v1-mock",
    inference_ms: 0,
  };
}
