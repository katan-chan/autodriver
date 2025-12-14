import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 8001,
    host: true, // Cho phép truy cập từ mọi IP (ví dụ từ mạng LAN hoặc ngrok)
    allowedHosts: ["*", "wired-lenient-treefrog.ngrok-free.app"], // Cho phép tất cả các host
    cors: true, // Cho phép mọi yêu cầu CORS
  },
});
