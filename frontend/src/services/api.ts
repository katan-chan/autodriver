import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://118.70.128.4:8000",
  headers: {
    "Content-Type": "application/json",
  },
});

export default api;
