import axios from "axios";
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Alert } from 'react-native';
import { uploadApi } from "./uploadApi";


// Axios 기본 설정
export const api = axios.create({
  baseURL: "http://ceprj.gachon.ac.kr:60013", // 백엔드 URL
  timeout: 30000,
});

// 요청 시 `access token`을 자동으로 포함시키기 위한 함수
export const getAuthHeaders = async () => {
  const accessToken = await AsyncStorage.getItem("access_token");
  if (accessToken) {
    return {
      Authorization: `Bearer ${accessToken}`,
    };
  }
  return {}; // access_token이 없으면 빈 객체 반환
};

// 로그인 API
export async function loginUser(email: string, password: string) {
  try {
    const response = await api.post('/api/users/login', {
      email,
      pwd: password,
    });

    console.log("🔥 백엔드 로그인 응답:", response.data);

    const access_token = response.data.accessToken;
    const refresh_token = response.data.refreshToken;
    const userId = response.data.userID;

    if (!access_token) throw new Error("백엔드에서 access_token을 반환하지 않았습니다.");

    // 🔥 올바르게 저장하기
    await AsyncStorage.setItem("access_token", access_token);
    await AsyncStorage.setItem("refresh_token", refresh_token);
    await AsyncStorage.setItem("user_id", userId.toString());

    return response.data;

  } catch (error: any) {
    console.log("❌ [API ERROR loginUser]");
    console.log("❗ status:", error.response?.status);
    console.log("❗ data:", error.response?.data);
    console.log("❗ message:", error.message);
    throw error;
  }
}


export async function registerUser(payload: any) {
  try {
    const response = await api.post("/api/users/register", payload);
    return response.data;
  } catch (error: any) {
    console.error("회원가입 실패:", error);
    throw error;
  }
}

// 비밀번호 재설정 이메일 전송
export async function sendResetEmail(email: string) {
  const res = await api.post("/api/email/send", null, {
    params: { email },
  });
  return res.data;
}

// 이메일 인증 코드 검증
export async function verifyResetCode(email: string, code: string) {
  const res = await api.post("/api/email/verify", null, {
    params: { email, code },
  });
  return res.data;
}

// 비밀번호 재설정
export const resetPassword = async (email: string, newPassword: string) => {
  const payload = {
    email: email.trim(),
    newPassword: newPassword.trim(),
  };
  return api.post('/api/users/reset-password', payload);
};

// 프로필 정보 가져오기
export async function getUserProfile() {
  try {
    const authHeaders = await getAuthHeaders();

    const res = await api.get(`/api/mypage`, {
      headers: authHeaders,
    });

    return res.data;
  } catch (err) {
    console.error("프로필 조회 실패", err);
    throw err;
  }
}

// 프로필 이미지 업로드
export async function updateProfileImage(file: {
  uri: string;
  name: string;
  type: string;
}) {
  try {
    console.log("===== PROFILE IMAGE UPLOAD START =====");

    const authHeaders = await getAuthHeaders();

    const formData = new FormData();
    formData.append("file", {
      uri: file.uri,
      name: file.name,
      type: file.type,
    } as any);

    const res = await uploadApi.post(
      "/api/mypage/profile",
      formData,
      {
        headers: {
          ...authHeaders,
          // ❗ Content-Type 절대 지정하지 않음
        },
      }
    );

    console.log("✅ 업로드 성공:", res.data);
    return res.data;
  } catch (err) {
    console.error("프로필 이미지 업로드 실패", err);
    throw err;
  }
}



// 프로필 정보 업데이트
export async function updateUserProfile(
  userId: number,
  email: string,
  nickName: string,
  gender: string
) {
  const payload = { userId, email, nickName, gender };
  const authHeaders = await getAuthHeaders();

  try {
    const response = await api.post(`/api/mypage/update`, payload, {
      headers: authHeaders,
    });
    return response.data;
  } catch (error) {
    console.error("프로필 업데이트 실패:", error);
    throw error;
  }
}

// 친구 요청 조회
export async function getFriendRequests() {
  const token = await AsyncStorage.getItem("access_token");

  const response = await api.get("/api/friend/requests/list", {
    headers: { Authorization: `Bearer ${token}` },
  });

  return response.data;
}
// 친구 추가 요청 보내기
export const sendFriendRequest = async (email: string) => {
  const token = await AsyncStorage.getItem("access_token");

  const res = await api.post(
    `/api/friend/request?email=${email}`,
    {},
    { headers: { Authorization: `Bearer ${token}` } }
  );

  return res.data;
};

// 친구 요청 수락
export const acceptFriendRequest = async (fromUserId: number) => {
  const token = await AsyncStorage.getItem("access_token");

  const res = await api.post(
    `/api/friend/accept?fromUserId=${fromUserId}`,
    {},
    { headers: { Authorization: `Bearer ${token}` } }
  );

  return res.data;
};



// 친구 요청 거절
export const rejectFriendRequest = async (fromUserId: number) => {
  const token = await AsyncStorage.getItem("access_token");

  const res = await api.post(
    `/api/friend/request/delete?fromUserId=${fromUserId}`,
    {},
    { headers: { Authorization: `Bearer ${token}` } }
  );

  return res.data;
};

// 친구 요청 목록 조회 (POST /api/friend/requests/list)
export const getFriendRequestList = async (token: string) => {
  try {
    const res = await api.post("/api/friend/requests/list", {
      token: token   // request body
    });

    return res.data;  // 요청 목록 배열
  } catch (error) {
    console.error("친구 요청 리스트 조회 실패:", error);
    throw error;
  }
};

export async function getFriendList() {
  const token = await AsyncStorage.getItem("access_token");

  const response = await api.get("/api/friend/list", {
    headers: { Authorization: `Bearer ${token}` },
  });

  return response.data;
}

export async function getFriendCalendar(friendId: number, date: string) {
  const token = await AsyncStorage.getItem("access_token");

  console.log("📤 getFriendCalendar 요청 params:", { friendId, date });

  const response = await api.get("/api/friend/calendar", {
    headers: { Authorization: `Bearer ${token}` },
    params: { friendId, date },
  });

  // 🔥 핵심 로그
  console.log("📥 getFriendCalendar 응답 전체:", response.data);
  console.log("📥 diaries 필드:", response.data?.diaries);

  return response.data;
}

// 친구 삭제
export const removeFriend = async (friendId: number) => {
  const token = await AsyncStorage.getItem("access_token");

  const res = await api.post(
    `/api/friend/delete`,   // 친구 삭제 API 경로 (가정)
    { friendId: friendId },  // 요청 본문에 친구 ID 포함
    { headers: { Authorization: `Bearer ${token}` } }
  );

  return res.data;
};


// 홈 데이터 가져오기
export const getHomeData = async () => {
  try {
    const authHeaders = await getAuthHeaders();
    const response = await api.get(`/api/home`, {
      headers: authHeaders,
    });
    return response.data;
  } catch (error) {
    console.error('홈 데이터 가져오기 실패:', error);
    throw error;
  }
};

// 📌 내 캘린더 조회
export async function getMyCalendar(date: string) {
  const token = await AsyncStorage.getItem("access_token");
  const userIdStr = await AsyncStorage.getItem("user_id");

  if (!token || !userIdStr) {
    throw new Error("로그인 정보가 없습니다.");
  }

  const userId = Number(userIdStr);

  const response = await api.get("/api/calendar/get", {
    headers: { Authorization: `Bearer ${token}` },
    params: { userId, date },
  });

  return response.data;
}

export async function getMyPage() {
  const authHeaders = await getAuthHeaders();

  const response = await api.get("/api/mypage", {
    headers: authHeaders,
  });

  return response.data;
}


export async function completeBookReport(id: number) {
  const token = await AsyncStorage.getItem("access_token");
  if (!token) throw new Error("로그인 정보가 없습니다.");

  const response = await api.post(
    `/api/bookreport/${id}/complete`,
    {},
    { headers: { Authorization: `Bearer ${token}` } }
  );

  return response.data; // {}
}


export interface MyBookSession {
  sessionId: number;
  title: string;
  emotion: string;
  genre: string;
  status: string;           // 진행중/완료 등
  createdAt: string;
  recommendTitle: string;
}

export async function getMyBookSessions(): Promise<MyBookSession[]> {
  const token = await AsyncStorage.getItem("access_token");
  if (!token) throw new Error("로그인 정보가 없습니다.");

  const response = await api.get("/api/bookreport/me/books", {
    headers: { Authorization: `Bearer ${token}` },
  });

  return response.data;
}


export interface BookReportItem {
  id: number;
  title: string;
  content: string;
  genre: string;
  emotion: string;
  createdAt: string; // "2025-12-07"
}

export async function getBookReportList(): Promise<BookReportItem[]> {
  const token = await AsyncStorage.getItem("access_token");
  if (!token) throw new Error("로그인 정보가 없습니다.");

  const response = await api.get("/api/bookreport/list", {
    headers: { Authorization: `Bearer ${token}` },
  });

  return response.data;
}

export type WritingType = "diary" | "book"; // 백엔드에서 실제 사용하는 문자열에 맞춰서 수정

export interface WritingStartResponse {
  sessionId: number;
  question: string;
}

export async function startWriting(type: WritingType): Promise<WritingStartResponse> {
  const token = await AsyncStorage.getItem("access_token");
  const userIdStr = await AsyncStorage.getItem("user_id");

  if (!token || !userIdStr) throw new Error("로그인 정보가 없습니다.");
  const userId = Number(userIdStr);

  const response = await api.post(
    "/api/writing/start",
    { type, userId },
    { headers: { Authorization: `Bearer ${token}` } }
  );

  return response.data;
}

export interface WritingAnswerResponse {
  nextQuestion: string;
  emotion: string;
  finalize: boolean;
  currentIndex: number;
  totalQuestions: number;
}

export async function sendWritingAnswer(
  sessionId: number,
  answer: string
): Promise<WritingAnswerResponse> {
  const token = await AsyncStorage.getItem("access_token");
  if (!token) throw new Error("로그인 정보가 없습니다.");

  const response = await api.post(
    "/api/writing/answer",
    { sessionId, answer },
    { headers: { Authorization: `Bearer ${token}` } }
  );

  return response.data;
}

export interface WritingFeedbackResponse {
  sessionId: number;
  done: boolean;
  question: string;
}

export async function sendWritingFeedback(
  sessionId: number,
  satisfied: boolean,
  addN: number
): Promise<WritingFeedbackResponse> {
  const token = await AsyncStorage.getItem("access_token");
  if (!token) throw new Error("로그인 정보가 없습니다.");

  const response = await api.post(
    "/api/writing/feedback",
    { sessionId, satisfied, addN },
    { headers: { Authorization: `Bearer ${token}` } }
  );

  return response.data;
}

export interface WritingFinalizeResponse {
  sessionId: number;
  title: string;
  content: string;
  emotion: string;
  emotionCount: number;
  recommendTitle: string;
  recommendGenre: string;
  date: string; // "2025-12-07"
}

export async function finalizeWriting(sessionId: number) {
  const token = await AsyncStorage.getItem("access_token");
  if (!token) throw new Error("로그인 정보가 없습니다.");

  try {
    const response = await api.post(
      "/api/writing/finalize",
      { sessionId },
      { headers: { Authorization: `Bearer ${token}` } }
    );

    console.log("finalizeWriting 응답:", response.data);
    return response.data;
  } catch (error: unknown) {  // error를 unknown 타입으로 처리
    if (error instanceof Error) {
      // Error 객체인 경우
      console.error("finalize 오류:", error.message);
      Alert.alert("오류", "최종 결과를 불러오는 중 문제가 발생했습니다.");
    } else {
      // Error 객체가 아닌 경우
      console.error("알 수 없는 오류 발생:", error);
      Alert.alert("오류", "알 수 없는 오류가 발생했습니다.");
    }
    throw error;  // 에러를 다시 던져서 호출한 곳에서 처리할 수 있게 함
  }
}



export async function completeWriting(id: number) {
  const token = await AsyncStorage.getItem("access_token");
  if (!token) throw new Error("로그인 정보가 없습니다.");

  const response = await api.post(
    `/api/writing/${id}/complete`,
    {},
    { headers: { Authorization: `Bearer ${token}` } }
  );

  return response.data; // {}
}

