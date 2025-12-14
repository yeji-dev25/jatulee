import React, { useEffect, useState } from "react";
import { View, Text, ScrollView, TouchableOpacity, StyleSheet } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { getFriendCalendar } from "../../api/services";
import { globalStyles, colors } from "../../styles/globalStyles";

export default function FriendCalendarScreen() {
  const router = useRouter();
  const { friendId } = useLocalSearchParams();

  const [loading, setLoading] = useState(true);
  const [friendInfo, setFriendInfo] = useState<any>(null);
  const [diaries, setDiaries] = useState([]);

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

  if (loading) return <Text>로딩 중...</Text>;

  return (
    <ScrollView style={globalStyles.screen}>
      <View style={globalStyles.header}>
        <Text style={globalStyles.title}>
          {friendInfo.freindNickName}님의 자투리
        </Text>
      </View>

      <View style={styles.statsBox}>
        <Text style={styles.statItem}>일기: {friendInfo.countDiary}</Text>
        <Text style={styles.statItem}>독후감: {friendInfo.countBook}</Text>
        <Text style={styles.statItem}>총 기록: {friendInfo.totalNum}</Text>
      </View>

      <Text style={globalStyles.sectionTitle}>기록 목록</Text>

      {diaries.length === 0 ? (
        <Text style={globalStyles.emptyText}>작성된 기록이 없습니다.</Text>
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
            <Text style={styles.diaryTitle}>{item.title}</Text>
            <Text style={styles.diaryInfo}>감정: {item.emotion}</Text>
            <Text style={styles.diaryDate}>{item.createdAt}</Text>
          </TouchableOpacity>
        ))
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  statsBox: {
    backgroundColor: colors.light,
    padding: 15,
    borderRadius: 10,
    marginVertical: 15,
  },
  statItem: {
    fontSize: 16,
    marginBottom: 5,
  },
  diaryCard: {
    backgroundColor: colors.white,
    padding: 15,
    borderRadius: 12,
    marginBottom: 12,
  },
  diaryTitle: {
    fontSize: 16,
    fontWeight: "bold",
  },
  diaryInfo: {
    marginTop: 5,
    color: colors.primary,
  },
  diaryDate: {
    marginTop: 5,
    color: colors.gray,
  },
});
