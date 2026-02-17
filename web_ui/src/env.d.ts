/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_CHAT_API_BASE_URL?: string;
  readonly VITE_EMBEDDING_API_BASE_URL?: string;
  readonly VITE_PROXY_TARGET?: string;
  readonly VITE_CHAT_PROXY_TARGET?: string;
  readonly VITE_EMBEDDING_PROXY_TARGET?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
