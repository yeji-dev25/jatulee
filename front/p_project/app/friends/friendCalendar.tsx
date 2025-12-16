import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { getFriendCalendar } from "../../api/services";
import { globalStyles, colors } from "../../styles/globalStyles";

export default function FriendCalendarScreen() {
  const router = useRouter();
  const { friendId } = useLocalSearchParams();

  const [loading, setLoading] = useState(true);
  const [friendInfo, setFriendInfo] = useState<any>(null);
  const [diaries, setDiaries] = useState<any[]>([]);

  useEffect(() => {
    loadCalendar();
  }, []);

  const loadCalendar = async () => {
    try {
      const today = new Date().toISOString().slice(0, 10);
      const data = await getFriendCalendar(Number(friendId), today);

      setFriendInfo(data);
      setDiaries(data.diaries || []);
    } catch (err) {
      console.error("친구 캘린더 로드 실패:", err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Text style={styles.loadingText}>
        로딩 중...
      </Text>
    );
  }

  return (
    <ScrollView style={globalStyles.screen}>
      {/* 🔥 헤더 */}
      <View style={globalStyles.header}>
          <Text
           style={{
             fontFamily: 'SubTitleFont',
             fontSize: 24,
             color: colors.dark,
             marginBottom: 5,
           }}
         >
          {friendInfo.freindNickName}님의 자투리
         </Text>
      </View>

      {/* 통계 */}
      <View style={styles.statsBox}>
        <Text style={styles.statItem}>일기: {friendInfo.countDiary}</Text>
        <Text style={styles.statItem}>독후감: {friendInfo.countBook}</Text>
        <Text style={styles.statItem}>총 기록: {friendInfo.totalNum}</Text>
      </View>

      {/* 섹션 타이틀 */}
      <Text style={styles.sectionTitle}>기록 목록</Text>

      {diaries.length === 0 ? (
        <Text style={styles.emptyText}>작성된 기록이 없습니다.</Text>
      ) : (
        diaries.map((item: any) => (
          <TouchableOpacity
            key={item.id}
            style={styles.diaryCard}
            onPress={() =>
              router.push({
                pathname: "/diary/detail",
                params: { id: item.id.toString() },
              })
            }
          >
            <Text style={styles.diaryTitle}>
              {item.title}
            </Text>
            <Text style={styles.diaryInfo}>
              감정: {item.emotion}
            </Text>
            <Text style={styles.diaryDate}>
              {item.createdAt}
            </Text>
          </TouchableOpacity>
        ))
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  /* =========================
     Loading
  ========================= */
  loadingText: {
    fontFamily: "DefaultFont",
    textAlign: "center",
    marginTop: 50,
    fontSize: 16,
  },

  /* =========================
     Header
  ========================= */
  pageTitle: {
    fontFamily: "SubTitleFont",
    fontSize: 24,
    fontWeight: "700",
    color: colors.dark,
    marginBottom: 5,
  },

  /* =========================
     Stats
  ========================= */
  statsBox: {
    backgroundColor: colors.light,
    padding: 15,
    borderRadius: 10,
    marginVertical: 15,
  },
  statItem: {
    fontFamily: "DefaultFont",
    fontSize: 16,
    marginBottom: 5,
    color: colors.dark,
  },

  /* =========================
     Section
  ========================= */
  sectionTitle: {
    fontFamily: "SubTitleFont",
    fontSize: 18,
    fontWeight: "600",
    color: colors.dark,
    marginBottom: 10,
  },
  emptyText: {
    fontFamily: "DefaultFont",
    fontSize: 14,
    color: colors.gray,
    textAlign: "center",
    marginTop: 20,
  },

  /* =========================
     Diary Card
  ========================= */
  diaryCard: {
    backgroundColor: colors.white,
    padding: 15,
    borderRadius: 12,
    marginBottom: 12,
    elevation: 2,
  },
  diaryTitle: {
    fontFamily: "TitleFont",
    fontSize: 16,
    color: colors.dark,
    marginBottom: 4,
  },
  diaryInfo: {
    fontFamily: "DefaultFont",
    marginTop: 5,
    color: colors.primary,
  },
  diaryDate: {
    fontFamily: "DefaultFont",
    marginTop: 5,
    color: colors.gray,
    fontSize: 12,
  },
});
