import React, { useState, useEffect } from "react";
import { useRouter } from "expo-router";
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
} from "react-native";
import { globalStyles, colors } from "../styles/globalStyles";
import { getBookReportList } from "../api/services"; //🔥 완성된 독후감 API

export default function BookReviewList() {
  const [reviews, setReviews] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    loadReviews();
  }, []);

  const loadReviews = async () => {
    try {
      const res = await getBookReportList();
      setReviews(res);
      console.log("📘 getBookReportList 결과:", res);
    } catch (e) {
      console.error("독후감 목록 불러오기 실패:", e);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <View style={globalStyles.screen}>
      <ScrollView style={globalStyles.scrollView}>
        {isLoading ? (
          <Text style={{ fontFamily: "DefaultFont" }}>
            로딩 중...
          </Text>
        ) : reviews.length === 0 ? (
          <Text style={{ fontFamily: "DefaultFont" }}>
            독후감이 없습니다.
          </Text>
        ) : (
          reviews.map((review: any) => {
            return (
              <TouchableOpacity
                key={review.id}
                style={styles.reviewCard}
                activeOpacity={0.8}
                onPress={() =>
                  router.push({
                    pathname: "/diary/detail",
                    params: {
                      id: review.id,
                      type: "book",
                    },
                  })
                }
              >
                {/* 제목 */}
                <Text style={styles.reviewTitle}>
                  {review.title}
                </Text>

                {/* 생성일 */}
                <Text style={styles.dateText}>
                  작성일: {review.createdAt?.slice(0, 10)}
                </Text>

                {/* 감정 */}
                {review.emotion && (
                  <Text style={styles.emotionText}>
                    감정: {review.emotion}
                  </Text>
                )}

                {/* 장르 */}
                {review.genre && (
                  <Text style={styles.genreText}>
                    장르: {review.genre}
                  </Text>
                )}

                {/* 추천 책 제목 */}
                {review.recommendTitle && (
                  <View style={styles.recommendationContainer}>
                    <Text style={styles.recommendationText}>
                      📚 추천 도서: "{review.recommendTitle}"
                    </Text>
                  </View>
                )}
              </TouchableOpacity>
            );
          })
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  reviewCard: {
    backgroundColor: colors.white,
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 3,
  },
  reviewTitle: {
    fontFamily: "TitleFont",        // 🔥
    fontSize: 18,
    fontWeight: "bold",
    color: colors.dark,
    marginBottom: 6,
  },
  dateText: {
    fontFamily: "DefaultFont",      // 🔥
    fontSize: 13,
    color: colors.gray,
    marginBottom: 8,
  },
  emotionText: {
    fontFamily: "SubTitleFont",     // 🔥
    fontSize: 14,
    color: colors.primary,
    marginBottom: 4,
  },
  genreText: {
    fontFamily: "DefaultFont",      // 🔥
    fontSize: 14,
    color: colors.dark,
    marginBottom: 6,
  },
  recommendationContainer: {
    backgroundColor: colors.primary + "20",
    padding: 12,
    borderRadius: 8,
    marginTop: 10,
  },
  recommendationText: {
    fontFamily: "SubTitleFont",     // 🔥
    fontSize: 14,
    color: colors.dark,
    fontWeight: "600",
  },
});
