import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Fail loudly if 5173 is taken, instead of silently picking 5174/5175.
    // The backend's CORS allowlist (see backend/main.py's DEFAULT_DEV_ORIGINS)
    // only trusts localhost:5173 by default -- if Vite silently moved to a
    // different port, every fetch() from the frontend gets blocked by CORS,
    // which looks like "everything is blank / failed to load" with no
    // obvious cause. Free up port 5173 (or set CORS_ORIGINS on the backend
    // to match whatever port Vite actually uses) rather than letting this
    // fail silently.
    strictPort: true,
  },
})
