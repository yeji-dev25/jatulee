// app/diary/write.tsx - AI 기반 일기/독후감 작성 화면 (API 연동 버전)

import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import { useRouter, useLocalSearchParams } from "expo-router";
import { globalStyles, colors } from "../../styles/globalStyles";

import {
  startWriting,
  sendWritingAnswer,
  sendWritingFeedback,
  finalizeWriting,
  WritingType,
  WritingFinalizeResponse,
} from "../../api/services";

export default function WriteScreen() {
  const router = useRouter();
  const rawParams = useLocalSearchParams();

  // 🔥 파라미터 로그 (디버깅용)
  useEffect(() => {
    console.log("🔥 받은 params:", rawParams);
  }, [rawParams]);

  /** -----------------------------
   *  params 안전하게 변환하기
   *  expo-router의 params는 string | string[] | undefined 가능
   ----------------------------- */
  const normalize = (value: string | string[] | undefined): string => {
    if (Array.isArray(value)) return value[0];
    return value ?? "";
  };

  const dateParam = normalize(rawParams.date as string | string[] | undefined);
  const displayDateParam = normalize(
    rawParams.displayDate as string | string[] | undefined
  );

  const selectedDate = dateParam
    ? {
        dateString: dateParam,
        displayDate: decodeURIComponent(displayDateParam || dateParam),
      }
    : null;

  /** ----------------------------- */

  const [diaryType, setDiaryType] = useState<WritingType>("diary");
  const [sessionId, setSessionId] = useState<number | null>(null);

  const [currentQuestion, setCurrentQuestion] = useState<string>("");
  const [currentAnswer, setCurrentAnswer] = useState<string>("");
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [totalQuestions, setTotalQuestions] = useState<number>(0);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isCompleted, setIsCompleted] = useState<boolean>(false);
  const [finalData, setFinalData] = useState<WritingFinalizeResponse | null>(
    null
  );

  // ============================================
  // 📌 1) 첫 질문 요청 startWriting()
  // ============================================
const startWritingSession = async () => {
  try {
    setIsLoading(true);
    console.log("로딩 시작");

    const res = await startWriting(diaryType);
    console.log("🔥 [startWriting 응답] >>>", res);

    const { sessionId, question } = res;

    if (!sessionId || !question) {
      throw new Error("writing/start 응답에 sessionId 또는 question 없음");
    }

    setSessionId(sessionId);
    setCurrentQuestion(question);
    setCurrentIndex(1);
    setTotalQuestions(5);
  } catch (err) {
    console.error("🔥 startWriting 처리 중 오류:", err);
    Alert.alert("오류", "AI 질문을 불러올 수 없습니다.");
  } finally {
    setIsLoading(false); // 로딩 종료
  }
};

useEffect(() => {
  console.log("useEffect 호출");
  startWritingSession();
}, [diaryType]);
  // ============================================
  // 📌 2) 답변 전송 → 다음 질문 받기
  // ============================================
const handleSendAnswer = async () => {
  if (!currentAnswer.trim()) {
    Alert.alert("알림", "답변을 입력해주세요.");
    return;
  }
  if (!sessionId) return;

  try {
    setIsLoading(true);
    const res = await sendWritingAnswer(sessionId, currentAnswer.trim());

    console.log("🔥 [sendWritingAnswer 응답] >>>", res);

    if (res.finalize) {
      await handleFeedback(); // 마지막 질문은 피드백 단계로 넘어감
      return;
    }

    // 다음 질문 세팅
    setCurrentQuestion(res.nextQuestion);
    setCurrentIndex(res.currentIndex + 1);
    setTotalQuestions(res.totalQuestions ?? totalQuestions);
    setCurrentAnswer(""); // 답변 초기화
  } catch (err) {
    console.error("답변 전송 오류:", err);
    Alert.alert("오류", "답변 전송 중 문제가 발생했습니다.");
  } finally {
    setIsLoading(false); // 로딩 종료
  }
};

  // ============================================
  // 📌 3) 만족도(feedback) 단계
  // ============================================
  const handleFeedback = async () => {
    if (!sessionId) return;

    try {
      const res = await sendWritingFeedback(sessionId, true, 0);
      console.log("🔥 [sendWritingFeedback 응답] >>>", res);

      if (res.done) {
        await finalizeSession();
      } else {
        setCurrentQuestion(res.question);
        setCurrentAnswer("");
      }
    } catch (err) {
      console.error("feedback 에러:", err);
      Alert.alert("오류", "피드백 처리 중 문제가 발생했습니다.");
    }
  };

  // ============================================
  // 📌 4) finalize → 최종 결과 받아오기
  // ============================================
  const finalizeSession = async () => {
    if (!sessionId) {
      console.error("sessionId 없음 → finalize 불가");
      return;
    }

    try {
      const res = await finalizeWriting(sessionId);
      console.log("🔥 [finalizeWriting 응답] >>>", res);
      setFinalData(res);
      setIsCompleted(true);
    } catch (err) {
      console.error("finalize 오류:", err);
      Alert.alert("오류", "최종 결과를 불러오는 중 문제가 발생했습니다.");
    }
  };

  // ============================================
  // 📌 로딩 화면
  // ============================================
  if (isLoading && !isCompleted) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={colors.primary} />
        <Text style={styles.loadingText}>불러오는 중...</Text>
      </View>
    );
  }

  // ============================================
  // 📌 5) 최종 결과 화면
  // ============================================
  if (isCompleted && finalData) {
    return (
      <View style={globalStyles.container}>
        <View style={globalStyles.header}>
          <Text style={globalStyles.title}>✅ 완성되었습니다!</Text>
          {selectedDate && (
            <Text style={globalStyles.subtitle}>{selectedDate.displayDate}</Text>
          )}
        </View>

        <ScrollView style={styles.generatedDiary}>
          <Text style={styles.diaryContent}>{finalData.content}</Text>

           {/* 공감 인원 표시 */}
  {typeof finalData.emotionCount === "number" && (
    <View style={styles.emotionCountBox}>
      <Text style={styles.emotionCountText}>
         나와 같은 감정을 느낀 사람은 {finalData.emotionCount}명입니다
      </Text>
    </View>
  )}

          {/* 추천 배너 */}
          <View style={styles.banner}>
            <Text style={styles.bannerIcon}>🤖</Text>
            <Text style={styles.bannerText}>
              {diaryType === "diary"
                ? `추천 노래: ${finalData.recommendTitle}`
                : `추천 책: ${finalData.recommendTitle}`}
            </Text>
          </View>
        </ScrollView>

        <View style={globalStyles.buttonContainer}>
          <TouchableOpacity
            style={[globalStyles.button, globalStyles.primaryButton]}
            onPress={() => router.replace("/calendar")}
          >
            <Text style={globalStyles.buttonText}>캘린더로 돌아가기</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // ============================================
  // 📌 6) 질문 입력 UI 화면
  // ============================================
  return (
    <View style={globalStyles.screen}>
      <View style={globalStyles.header}>
        <Text style={globalStyles.title}>
          AI 질문
        </Text>

        {selectedDate && (
          <Text style={globalStyles.subtitle}>{selectedDate.displayDate}</Text>
        )}

        {/* 타입 선택 버튼 */}
        <View style={styles.typeSelector}>
          <TouchableOpacity
            style={[styles.typeButton, diaryType === "diary" && styles.activeTypeButton]}
            onPress={() => setDiaryType("diary")}
          >
            <Text style={[styles.typeButtonText, diaryType === "diary" && styles.activeTypeButtonText]}>
              일기
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.typeButton, diaryType === "book" && styles.activeTypeButton]}
            onPress={() => setDiaryType("book")}
          >
            <Text style={[styles.typeButtonText, diaryType === "book" && styles.activeTypeButtonText]}>
              독후감
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* 질문 텍스트 */}
      <View style={styles.questionContainer}>
        <Text style={styles.questionText}>
          {currentQuestion || "AI가 질문을 준비 중입니다..."}
        </Text>
      </View>

      {/* 답변 입력 */}
      <View style={styles.answerContainer}>
        <TextInput
          style={styles.answerInput}
          placeholder="답변을 입력하세요..."
          multiline
          value={currentAnswer}
          onChangeText={setCurrentAnswer}
        />
      </View>

      <View style={globalStyles.buttonContainer}>
        <TouchableOpacity
          style={[globalStyles.button, globalStyles.primaryButton]}
          onPress={handleSendAnswer}
        >
          <Text style={globalStyles.buttonText}>다음</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "transparent", // 배경을 투명하게 설정
  },
  loadingText: {
    marginTop: 10,
    color: colors.primary,  // 로딩 텍스트 색상 설정
    fontSize: 16,
  },
  typeSelector: {
    flexDirection: "row" as const,
    marginTop: 15,
    backgroundColor: colors.light,
    borderRadius: 8,
    padding: 2,
  },
  typeButton: {
    flex: 1,
    paddingVertical: 8,
    alignItems: "center" as const,
    borderRadius: 6,
  },
  activeTypeButton: {
    backgroundColor: colors.primary,
  },
  typeButtonText: {
    fontSize: 14,
    color: colors.gray,
  },
  activeTypeButtonText: {
    color: colors.white,
    fontWeight: "600" as const,
  },
  questionContainer: {
    backgroundColor: colors.white,
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
  },
  questionText: {
    fontSize: 18,
    color: colors.dark,
    textAlign: "center" as const,
  },
  answerContainer: {
    flex: 1,
    marginBottom: 20,
  },
  emotionCountBox: {
  marginTop: 20,
  paddingVertical: 14,
  paddingHorizontal: 16,
  backgroundColor: "#E3F2FD",
  borderRadius: 12,
  alignItems: "center",
},

emotionCountText: {
  fontSize: 14,
  fontWeight: "600",
  color: colors.primary,
},
  answerInput: {
    backgroundColor: colors.white,
    borderRadius: 8,
    padding: 15,
    fontSize: 16,
    minHeight: 150,
    borderWidth: 1,
    borderColor: colors.lightGray,
  },
  generatedDiary: {
    backgroundColor: colors.white,
    padding: 20,
    margin: 20,
    borderRadius: 12,
  },
  diaryContent: {
    fontSize: 16,
    lineHeight: 24,
    color: colors.dark,
  },
  banner: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    backgroundColor: colors.light,
    padding: 15,
    borderRadius: 12,
    marginTop: 20,
  },
  bannerIcon: {
    fontSize: 24,
    marginRight: 10,
  },
  bannerText: {
    fontSize: 14,
    color: colors.dark,
  },
});
