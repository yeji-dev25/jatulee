import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Alert,
} from "react-native";
import { useRouter } from "expo-router";
import { getHomeData } from "../../api/services";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { globalStyles, colors } from "../../styles/globalStyles";

interface User {
  id: number;
  username: string;
  name: string;
}

interface Diary {
  id: number | undefined;
  title: string | null;
  content: string | null;
  emotion: string | null;
  date: string;
  dateString: string;
  type: "diary" | "book";
}

export default function HomeScreen() {
  const router = useRouter();

  const [user, setUser] = useState<User | null>(null);
  const [diaries, setDiaries] = useState<Diary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const showDetailDisabledAlert = () => {
    Alert.alert(
      "상세 보기 준비 중",
      "현재 이 기록은 상세 보기를 지원하지 않습니다.\n추후 업데이트를 기다려주세요."
    );
  };

  const loadData = async () => {
    try {
      const token = await AsyncStorage.getItem("access_token");
      if (!token) {
        Alert.alert("알림", "로그인이 필요합니다.");
        return;
      }

      const data = await getHomeData();

      const userData = {
        id: data.userId,
        username: data.nickName,
        name: data.name,
      };

      setUser(userData);
      await AsyncStorage.setItem("user", JSON.stringify(userData));

      const rawDiaries = data.writingSessionDTOS || [];

      const mappedDiaries: Diary[] = rawDiaries.map((item: any) => ({
        id: item.id ?? item.writingSessionId,
        title: item.title ?? null,
        content: item.content ?? null,
        emotion: item.emotion ?? null,
        date: item.createdAt ?? "",
        dateString: (item.createdAt ?? "").slice(0, 10),
        type: item.type ?? "diary",
      }));

      setDiaries(mappedDiaries);
    } catch (err) {
      Alert.alert("오류", "데이터를 불러오는 중 문제가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <View style={globalStyles.center}>
        <Text style={globalStyles.text}>로딩 중...</Text>
      </View>
    );
  }

  const today = new Date().toISOString().slice(0, 10);
  const todayDiary = diaries.find((d) => d.dateString === today);
  const recentDiaries = diaries.slice(0, 5);

  const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
  const thisWeekCount = diaries.filter(
    (d) => new Date(d.dateString) >= weekAgo
  ).length;

  return (
    <ScrollView style={globalStyles.screen}>
      {/* 헤더 */}
      <View style={globalStyles.header}>
        <Text style={styles.greeting}>
          안녕하세요, {user?.username}님!
        </Text>
      </View>

      {/* 오늘의 기록 */}
      <View style={styles.todaySection}>
        <Text style={styles.sectionTitle}>오늘의 기록</Text>

        {todayDiary ? (
          <TouchableOpacity
            style={styles.todayCard}
            onPress={showDetailDisabledAlert}
          >
            <Text style={styles.todayCardTitle}>
              오늘의 기록이 있습니다
            </Text>
            <Text style={styles.todayCardEmotion}>
              감정: {todayDiary.emotion ?? "알 수 없음"}
            </Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={styles.todayEmptyCard}
            onPress={() => router.push("/diary/write")}
          >
            <Text style={styles.todayEmptyText}>
              오늘의 일기가 없습니다
            </Text>
            <Text style={styles.todayEmptySubtext}>
              탭해서 작성하기
            </Text>
          </TouchableOpacity>
        )}
      </View>

      {/* 이번 주 활동 */}
      <View style={globalStyles.card}>
        <Text style={styles.sectionTitle}>이번 주 활동</Text>

        <View style={styles.statsRow}>
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>{thisWeekCount}</Text>
            <Text style={styles.statLabel}>개 기록</Text>
          </View>

          <View style={styles.statItem}>
            <Text style={styles.statNumber}>{diaries.length}</Text>
            <Text style={styles.statLabel}>총 기록</Text>
          </View>
        </View>
      </View>

      {/* 최근 기록 */}
      <View style={styles.recentSection}>
        <Text style={styles.sectionTitle}>최근 기록</Text>

        {recentDiaries.length === 0 ? (
          <Text style={globalStyles.emptyText}>
            아직 작성된 기록이 없습니다.
          </Text>
        ) : (
          recentDiaries.map((diary, index) => (
            <TouchableOpacity
              key={`recent-${index}`}
              style={globalStyles.listItem}
              onPress={showDetailDisabledAlert}
            >
              <View style={globalStyles.listItemHeader}>
                <Text style={styles.listTitle}>
                  {diary.title ?? `기록 ${index + 1}`}
                </Text>

                <Text style={styles.diaryType}>
                  {diary.type === "diary" ? "일기" : "독후감"}
                </Text>
              </View>

              <Text style={globalStyles.listItemSubtitle}>
                {diary.dateString}
              </Text>

              <Text style={styles.diaryEmotion}>
                감정: {diary.emotion ?? "없음"}
              </Text>
            </TouchableOpacity>
          ))
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  greeting: {
    fontFamily: "TitleFont",
    fontSize: 24,
    color: colors.dark,
  },

  sectionTitle: {
    fontFamily: "TitleFont",
    fontSize: 18,
    marginBottom: 12,
    color: colors.dark,
  },

  todaySection: { marginBottom: 25 },

  todayCard: {
    backgroundColor: colors.primary,
    padding: 20,
    borderRadius: 12,
    alignItems: "center",
  },
  todayCardTitle: {
    fontFamily: "TitleFont",
    fontSize: 18,
    color: colors.white,
  },
  todayCardEmotion: {
    fontFamily: "DefaultFont",
    color: colors.white,
    marginTop: 6,
  },

  todayEmptyCard: {
    backgroundColor: colors.light,
    padding: 20,
    borderRadius: 12,
    alignItems: "center",
    borderWidth: 2,
    borderColor: colors.lightGray,
    borderStyle: "dashed",
  },
  todayEmptyText: {
    fontFamily: "SubTitleFont",
    color: colors.gray,
    fontSize: 16,
  },
  todayEmptySubtext: {
    fontFamily: "DefaultFont",
    color: colors.gray,
    fontSize: 12,
  },

  statsRow: {
    flexDirection: "row",
    justifyContent: "space-around",
  },
  statItem: { alignItems: "center" },
  statNumber: {
    fontFamily: "TitleFont",
    fontSize: 24,
    color: colors.primary,
  },
  statLabel: {
    fontFamily: "DefaultFont",
    fontSize: 12,
    color: colors.gray,
  },

  recentSection: { marginBottom: 25 },

  listTitle: {
    fontFamily: "SubTitleFont",
    fontSize: 16,
    color: colors.dark,
  },

  diaryType: {
    fontFamily: "DefaultFont",
    fontSize: 12,
    color: colors.primary,
    backgroundColor: "#E3F2FD",
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },

  diaryEmotion: {
    fontFamily: "DefaultFont",
    fontSize: 14,
    color: colors.primary,
  },
});
