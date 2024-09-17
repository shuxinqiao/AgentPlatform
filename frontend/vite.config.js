import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  server: {
    proxy: {
      "^/db/.*": {
        target: "http://localhost:8000",
        changeOrigin: true,
        // rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "^/ws/chat/.*": {
        target: "ws://localhost:8000",
        ws: true,
        rewriteWsOrigin: true,
      },
      "^/temp/.*": {
        target: "ws://localhost:8000",
        changeOrigin: true,
      },
    },
  },
})

