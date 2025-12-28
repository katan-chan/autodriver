import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 8001,
    host: "0.0.0.0", // Cho phép truy cập từ mọi IP (ví dụ từ mạng LAN hoặc ngrok)
    strictPort: false,
    cors: true, // Cho phép mọi yêu cầu CORS
  },
});
