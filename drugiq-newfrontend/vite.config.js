import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev-only proxy so `npm run dev` (Vite on :5173) can call the Flask API
// on :5002 without CORS setup. Production is served straight from Flask
// (see app.py), where the API is already same-origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:5002',
    },
  },
})
