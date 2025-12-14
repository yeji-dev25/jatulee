// app/diary/detail.tsx

import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import { useRouter, useLocalSearchParams } from "expo-router";
import { globalStyles, colors } from "../../styles/globalStyles";
import { getBookReportList } from "../../api/services";

// ==========================
// 📌 Diary 타입 정의
// ==========================
interface DiaryItem {
  id: number;
  title: string;
  content: string;
  emotion?: string;
  date?: string;
  createdAt?: string;
  type: "diary" | "book";
  genre?: string | null;
  author?: string | null;
}

export default function DiaryDetailScreen() {
  const router = useRouter();
  const params = useLocalSearchParams();

  const [diary, setDiary] = useState<DiaryItem | null>(null);
  const [loading, setLoading] = useState(true);

  // ================================
  // 🔥 params 안전하게 변환
  // ================================
  const normalize = (v: string | string[] | undefined): string => {
    if (Array.isArray(v)) return v[0];
    return v ?? "";
  };

  const idParam = normalize(params.id);
  const typeParam = normalize(params.type);
  const itemParam = normalize(params.item);

  // ================================
  // 📌 상세 로딩
  // ================================
  useEffect(() => {
    const loadDetail = async () => {
      try {
        // 🔥 case 1: 리스트에서 item 전체를 넘겨준 경우
        if (itemParam) {
          const parsed = JSON.parse(itemParam);
          setDiary(parsed);
          return;
        }

        // 🔥 case 2: 독후감 상세 → API에서 조회
        if (idParam && typeParam === "book") {
          const list = await getBookReportList();
          const found = list.find((b: any) => b.id === Number(idParam));

          if (found) {
            setDiary({
              id: found.id,
              title: found.title,
              content: found.content,
              emotion: found.emotion,
              genre: found.genre,
              type: "book",
              createdAt: found.createdAt,
            });
          }
        }
      } catch (err) {
        console.error("상세 조회 실패:", err);
      } finally {
        setLoading(false);
      }
    };

    loadDetail();
  }, []);

  // ================================
  // 📌 로딩 화면 (로딩은 질문 화면에서만 보이게)
  // ================================
  if (loading) {
    return (
      <View style={globalStyles.center}>
        <ActivityIndicator size="large" color={colors.primary} />
        <Text style={styles.loadingText}>불러오는 중...</Text>
      </View>
    );
  }

  // ================================
  // 📌 잘못된 접근
  // ================================
  if (!diary) {
    return (
      <View style={globalStyles.screen}>
        <Text style={globalStyles.emptyText}>일기/독후감을 찾을 수 없습니다.</Text>
      </View>
    );
  }

  // ================================
  // 📌 상세 화면
  // ================================
  return (
    <View style={globalStyles.screen}>
      {/* 헤더 */}
      <View style={globalStyles.header}>
        <Text style={globalStyles.title}>{diary.title}</Text>
        <Text style={globalStyles.subtitle}>
          {diary.createdAt ?? diary.date ?? ""}
        </Text>
      </View>

      <ScrollView style={globalStyles.scrollView}>
        {/* 메타 정보 */}
        <View style={styles.metaContainer}>
          <View style={styles.typeRow}>
            <View style={styles.typeBadge}>
              <Text style={styles.typeText}>
                {diary.type === "diary" ? "📝 일기" : "📚 독후감"}
              </Text>
            </View>
          </View>

          {diary.emotion && (
            <View style={styles.emotionBadge}>
              <Text style={styles.emotionText}>{diary.emotion}</Text>
            </View>
          )}
        </View>

        {/* 독후감 장르 */}
        {diary.type === "book" && diary.genre && (
          <View style={styles.bookMeta}>
            <View style={styles.genreBadge}>
              <Text style={styles.genreText}>장르: {diary.genre}</Text>
            </View>
          </View>
        )}

        {/* 본문 */}
        <View style={styles.contentContainer}>
          <Text style={styles.contentText}>{diary.content}</Text>
        </View>
      </ScrollView>

      {/* 뒤로 버튼 */}
      <View style={styles.actionContainer}>
        <TouchableOpacity
          style={[globalStyles.button, globalStyles.secondaryButton]}
          onPress={() => router.back()}
        >
          <Text style={globalStyles.secondaryButtonText}>뒤로</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ==========================
// 📌 스타일 (TS 호환)
// ==========================
const styles = StyleSheet.create({
  metaContainer: {
    backgroundColor: colors.white,
    padding: 15,
    borderRadius: 12,
    marginBottom: 15,
  },
  typeRow: {
    flexDirection: "row",
    marginBottom: 10,
  },
  typeBadge: {
    backgroundColor: colors.primary,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  typeText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 12,
  },
  emotionBadge: {
    alignSelf: "flex-start",
    backgroundColor: colors.secondary,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  emotionText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 14,
  },
  bookMeta: {
    backgroundColor: colors.white,
    padding: 15,
    borderRadius: 12,
    marginBottom: 15,
  },
  genreBadge: {
    backgroundColor: colors.warning,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  genreText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 12,
  },
  contentContainer: {
    backgroundColor: colors.white,
    padding: 20,
    borderRadius: 12,
  },
  contentText: {
    fontSize: 16,
    lineHeight: 24,
    color: colors.dark,
  },
  actionContainer: {
    paddingVertical: 15,
    borderTopWidth: 1,
    borderTopColor: colors.lightGray,
    alignItems: "center",
  },
  loadingText: {
    marginTop: 10,
    color: colors.primary,
    fontSize: 16,
  },
});
