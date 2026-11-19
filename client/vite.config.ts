import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

// `npm run mock` (режим mock) собирает клиент на заглушках: стенд не трогается, лимит нейросети не тратится.
// base "./" — сборка работает из любой папки сервера, а не только из корня домена.
export default defineConfig(({ mode }) => ({
  base: "./",
  plugins: [react()],
  define: { __MOCK__: JSON.stringify(mode === "mock") },
  server: { port: 5173, host: true },
  preview: { port: 5173, host: true },
}))
