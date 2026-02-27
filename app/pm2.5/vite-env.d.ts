/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_SUPABASE_URL: string;
  readonly VITE_SUPABASE_ANON_KEY: string;
  readonly VITE_HF_INFERENCE_URL: string;
  readonly VITE_HF_API_TOKEN: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
