// app/diary/write.tsx

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
  Modal,
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

  const normalize = (value: string | string[] | undefined): string => {
    if (Array.isArray(value)) return value[0];
    return value ?? "";
  };

  const dateParam = normalize(rawParams.date as any);
  const displayDateParam = normalize(rawParams.displayDate as any);

  const selectedDate = dateParam
    ? {
        dateString: dateParam,
        displayDate: decodeURIComponent(displayDateParam || dateParam),
      }
    : null;

  const [diaryType, setDiaryType] = useState<WritingType>("diary");
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState("");
  const [currentAnswer, setCurrentAnswer] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  // 🔥 추가
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);

  const [isCompleted, setIsCompleted] = useState(false);
  const [finalData, setFinalData] =
    useState<WritingFinalizeResponse | null>(null);

  const startWritingSession = async () => {
    try {
      setIsLoading(true);
      const res = await startWriting(diaryType);
      setSessionId(res.sessionId);
      setCurrentQuestion(res.question);
      setCurrentIndex(1);
      setTotalQuestions(5);
    } catch {
      Alert.alert("오류", "AI 질문을 불러올 수 없습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    startWritingSession();
  }, [diaryType]);

  const handleSendAnswer = async () => {
    if (!currentAnswer.trim() || !sessionId) return;

    try {
      setIsLoading(true);
      const res = await sendWritingAnswer(sessionId, currentAnswer.trim());

      if (res.finalize) {
        await handleFeedback();
        return;
      }

      setCurrentQuestion(res.nextQuestion);
      setCurrentIndex(res.currentIndex + 1);
      setTotalQuestions(res.totalQuestions ?? totalQuestions);
      setCurrentAnswer("");
    } catch {
      Alert.alert("오류", "답변 전송 중 문제가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedback = async () => {
    if (!sessionId) return;
    const res = await sendWritingFeedback(sessionId, true, 0);
    if (res.done) finalizeSession();
    else {
      setCurrentQuestion(res.question);
      setCurrentAnswer("");
    }
  };

  const finalizeSession = async () => {
    if (!sessionId) return;
    const res = await finalizeWriting(sessionId);
    setFinalData(res);
    setIsCompleted(true);
  };

  // 🔥 완료 후 5초 뒤 만족도 모달
  useEffect(() => {
    if (isCompleted && finalData) {
      const timer = setTimeout(() => {
        setShowFeedbackModal(true);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [isCompleted, finalData]);

  // 🔥 만족 / 불만족 처리
  const handleSatisfaction = async (satisfied: boolean) => {
    if (!sessionId) return;

    try {
      setShowFeedbackModal(false);

      const addN = satisfied ? 0 : 2;
      const res = await sendWritingFeedback(sessionId, satisfied, addN);

      if (res.done) return;

      // ❌ 불만족 → 다시 질문 화면
      setIsCompleted(false);
      setFinalData(null);
      setCurrentQuestion(res.question);
      setCurrentAnswer("");
      setCurrentIndex(1);
      setTotalQuestions(addN);
    } catch {
      Alert.alert("오류", "만족도 처리 중 문제가 발생했습니다.");
    }
  };

  /* ================= 로딩 ================= */
  if (isLoading && !isCompleted) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={colors.primary} />
        <Text style={styles.loadingText}>불러오는 중...</Text>
      </View>
    );
  }

  /* ================= 완료 ================= */
  if (isCompleted && finalData) {
    return (
      <View style={globalStyles.container}>
        <View style={globalStyles.header}>
          <Text style={[globalStyles.title, { fontFamily: "SubTitleFont" }]}>
            ✅ 완성되었습니다!
          </Text>

          {selectedDate && (
            <Text
              style={[
                globalStyles.subtitle,
                { fontFamily: "DefaultFont" },
              ]}
            >
              {selectedDate.displayDate}
            </Text>
          )}
        </View>

        <ScrollView style={styles.generatedDiary}>
          <Text style={styles.diaryContent}>{finalData.content}</Text>

          {typeof finalData.emotionCount === "number" && (
            <View style={styles.emotionCountBox}>
              <Text style={styles.emotionCountText}>
                나와 같은 감정을 느낀 사람은 {finalData.emotionCount}명입니다
              </Text>
            </View>
          )}

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

        {/* 🔥 만족도 모달 */}
        <Modal transparent visible={showFeedbackModal} animationType="fade">
          <View style={styles.modalOverlay}>
            <View style={styles.modalContent}>
              <Text style={styles.modalTitle}>
                결과가 마음에 드셨나요?
              </Text>

              <View style={styles.modalButtons}>
                <TouchableOpacity
                  style={[styles.modalButton, styles.goodButton]}
                  onPress={() => handleSatisfaction(true)}
                >
                  <Text style={styles.modalButtonText}>😊 만족</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={[styles.modalButton, styles.badButton]}
                  onPress={() => handleSatisfaction(false)}
                >
                  <Text style={styles.modalButtonText}>😐 불만족</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>
      </View>
    );
  }

  /* ================= 작성 ================= */
  return (
    <View style={globalStyles.screen}>
      <View style={globalStyles.header}>
        <Text
          style={{
            fontFamily: "SubTitleFont",
            fontSize: 24,
            color: colors.dark,
            marginBottom: 5,
          }}
        >
          자투리 대화
        </Text>

        {selectedDate && (
          <Text
            style={[
              globalStyles.subtitle,
              { fontFamily: "DefaultFont" },
            ]}
          >
            {selectedDate.displayDate}
          </Text>
        )}

        <View style={styles.typeSelector}>
          <TouchableOpacity
            style={[
              styles.typeButton,
              diaryType === "diary" && styles.activeTypeButton,
            ]}
            onPress={() => setDiaryType("diary")}
          >
            <Text
              style={[
                styles.typeButtonText,
                diaryType === "diary" && styles.activeTypeButtonText,
              ]}
            >
              일기
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[
              styles.typeButton,
              diaryType === "book" && styles.activeTypeButton,
            ]}
            onPress={() => setDiaryType("book")}
          >
            <Text
              style={[
                styles.typeButtonText,
                diaryType === "book" && styles.activeTypeButtonText,
              ]}
            >
              독후감
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.questionContainer}>
        <Text style={styles.questionText}>{currentQuestion}</Text>
      </View>

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

/* ================= 스타일 ================= */

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: colors.primary,
    fontFamily: "DefaultFont",
  },
  typeSelector: {
    flexDirection: "row",
    marginTop: 15,
    backgroundColor: colors.light,
    borderRadius: 8,
    padding: 2,
  },
  typeButton: {
    flex: 1,
    paddingVertical: 8,
    alignItems: "center",
    borderRadius: 6,
  },
  activeTypeButton: {
    backgroundColor: colors.primary,
  },
  typeButtonText: {
    fontSize: 14,
    color: colors.gray,
    fontFamily: "DefaultFont",
  },
  activeTypeButtonText: {
    color: colors.white,
    fontWeight: "600",
    fontFamily: "SubTitleFont",
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
    textAlign: "center",
    fontFamily: "SubTitleFont",
  },
  answerContainer: {
    flex: 1,
    marginBottom: 20,
  },
  answerInput: {
    backgroundColor: colors.white,
    borderRadius: 8,
    padding: 15,
    fontSize: 16,
    minHeight: 150,
    borderWidth: 1,
    borderColor: colors.lightGray,
    fontFamily: "DefaultFont",
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
    fontFamily: "DefaultFont",
  },
  emotionCountBox: {
    marginTop: 20,
    padding: 14,
    backgroundColor: "#E3F2FD",
    borderRadius: 12,
    alignItems: "center",
  },
  emotionCountText: {
    fontSize: 14,
    fontWeight: "600",
    color: colors.primary,
    fontFamily: "SubTitleFont",
  },
  banner: {
    flexDirection: "row",
    alignItems: "center",
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
    fontFamily: "DefaultFont",
  },

  // 🔥 모달 스타일
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.4)",
    justifyContent: "center",
    alignItems: "center",
  },
  modalContent: {
    width: "80%",
    backgroundColor: colors.white,
    padding: 20,
    borderRadius: 12,
    alignItems: "center",
  },
  modalTitle: {
    fontSize: 18,
    fontFamily: "SubTitleFont",
    marginBottom: 20,
    color: colors.dark,
  },
  modalButtons: {
    flexDirection: "row",
    gap: 12,
  },
  modalButton: {
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  goodButton: {
    backgroundColor: colors.primary,
  },
  badButton: {
    backgroundColor: colors.gray,
  },
  modalButtonText: {
    color: colors.white,
    fontFamily: "DefaultFont",
    fontSize: 14,
  },
});
