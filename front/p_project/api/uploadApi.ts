import axios from "axios";

export const uploadApi = axios.create({
  baseURL: "http://ceprj.gachon.ac.kr:60013",
  timeout: 30000,
  // ❗ headers 절대 지정하지 말 것
});
