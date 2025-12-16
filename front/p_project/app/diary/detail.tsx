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

  const normalize = (v: string | string[] | undefined): string => {
    if (Array.isArray(v)) return v[0];
    return v ?? "";
  };

  const idParam = normalize(params.id);
  const typeParam = normalize(params.type);
  const itemParam = normalize(params.item);

  useEffect(() => {
    const loadDetail = async () => {
      try {
        if (itemParam) {
          const parsed = JSON.parse(itemParam);
          setDiary(parsed);
          return;
        }

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

  if (loading) {
    return (
      <View style={globalStyles.center}>
        <ActivityIndicator size="large" color={colors.primary} />
        <Text style={styles.loadingText}>불러오는 중...</Text>
      </View>
    );
  }

  if (!diary) {
    return (
      <View style={globalStyles.screen}>
        <Text style={globalStyles.emptyText}>
          일기/독후감을 찾을 수 없습니다.
        </Text>
      </View>
    );
  }

  return (
    <View style={globalStyles.screen}>
      {/* 헤더 */}
      <View style={globalStyles.header}>
        <Text
          style={[
            globalStyles.title,
            { fontFamily: "SubTitleFont" }, // 🔥
          ]}
        >
          {diary.title}
        </Text>
        <Text
          style={[
            globalStyles.subtitle,
            { fontFamily: "DefaultFont" }, // 🔥
          ]}
        >
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
              <Text style={styles.genreText}>
                장르: {diary.genre}
              </Text>
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
          <Text
            style={[
              globalStyles.secondaryButtonText,
              { fontFamily: "DefaultFont" }, // 🔥
            ]}
          >
            뒤로
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

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
    fontSize: 12,
    fontWeight: "600",
    fontFamily: "SubTitleFont", // 🔥
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
    fontSize: 14,
    fontWeight: "600",
    fontFamily: "SubTitleFont", // 🔥
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
    fontSize: 12,
    fontWeight: "600",
    fontFamily: "SubTitleFont", // 🔥
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
    fontFamily: "DefaultFont", // 🔥
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
    fontFamily: "DefaultFont", // 🔥
  },
});
