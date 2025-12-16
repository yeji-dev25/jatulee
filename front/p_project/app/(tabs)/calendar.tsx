// app/(tabs)/calendar.tsx
import React, { useState } from 'react';
import {
  Alert,
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Modal,
  Dimensions,
  StyleSheet,
} from 'react-native';
import { useRouter } from 'expo-router';
import { globalStyles, colors } from '../../styles/globalStyles';
import { getMyCalendar } from '../../api/services';

const { width: screenWidth } = Dimensions.get('window');

interface Diary {
  id: number;
  title: string;
  content: string;
  emotion: string;
  dateString: string;
  type: 'diary' | 'book_review';
  genre?: string | null;
  author: string;
  rating?: number | null;
  privacy: 'private' | 'friends' | 'public';
}

interface SelectedCalendarDate {
  day: number;
  dateString: string;
  displayDate: string;
  hasDiary: boolean;
  diary?: Diary;
}

export default function CalendarScreen() {
  const router = useRouter();
  const [showDateModal, setShowDateModal] = useState(false);
  const [selectedCalendarDate, setSelectedCalendarDate] =
    useState<SelectedCalendarDate | null>(null);
  const [currentDate, setCurrentDate] = useState(new Date());

  const currentYear = currentDate.getFullYear();
  const currentMonth = currentDate.getMonth();

  const today = new Date();
  const todayDate = today.getDate();
  const todayMonth = today.getMonth();
  const todayYear = today.getFullYear();

  const firstDay = new Date(currentYear, currentMonth, 1);
  const lastDay = new Date(currentYear, currentMonth + 1, 0);
  const daysInMonth = lastDay.getDate();
  const startDayOfWeek = firstDay.getDay();

  const calendarDays: (number | null)[] = [];
  for (let i = 0; i < startDayOfWeek; i++) calendarDays.push(null);
  for (let day = 1; day <= daysInMonth; day++) calendarDays.push(day);

  const handleDatePress = async (day: number) => {
    const dateString = `${currentYear}-${String(currentMonth + 1).padStart(
      2,
      '0'
    )}-${String(day).padStart(2, '0')}`;

    try {
      const data = await getMyCalendar(dateString);
      const diaryData = data.diaries?.length > 0 ? data.diaries[0] : null;

      setSelectedCalendarDate({
        day,
        dateString,
        displayDate: `${currentYear}년 ${currentMonth + 1}월 ${day}일`,
        hasDiary: !!diaryData,
        diary: diaryData
          ? {
              id: diaryData.id,
              title: diaryData.title,
              content: diaryData.content,
              emotion: diaryData.emotion,
              genre: diaryData.genre,
              dateString: diaryData.createdAt,
              type: 'diary',
              author: '',
              privacy: 'private',
            }
          : undefined,
      });

      setShowDateModal(true);
    } catch (error) {
      Alert.alert('오류', '해당 날짜 데이터를 불러올 수 없습니다.');
    }
  };

  const handleDateAction = (action: 'view' | 'write') => {
    setShowDateModal(false);

    if (action === 'view' && selectedCalendarDate?.diary) {
      router.push({
        pathname: '/diary/detail',
        params: { id: selectedCalendarDate.diary.id.toString() },
      } as any);
    }

    if (action === 'write' && selectedCalendarDate) {
      router.push({
        pathname: '/diary/write',
        params: {
          date: selectedCalendarDate.dateString,
          displayDate: selectedCalendarDate.displayDate,
        },
      });
    }
  };

  const changeMonth = (direction: 'prev' | 'next') => {
    const newDate = new Date(currentDate);
    newDate.setMonth(currentMonth + (direction === 'prev' ? -1 : 1));
    setCurrentDate(newDate);
  };

  return (
    <ScrollView style={globalStyles.screen}>
      {/* 헤더 */}
<View style={globalStyles.header}>
  <Text
    style={{
      fontFamily: 'SubTitleFont',
      fontSize: 24,
      color: colors.dark,
      marginBottom: 5,
    }}
  >
    캘린더
  </Text>

  <View style={styles.monthNavigation}>
    <TouchableOpacity onPress={() => changeMonth('prev')}>
      <Text style={styles.arrow}>◀</Text>
    </TouchableOpacity>

    {/* 🔥 연 / 월 폰트 분리 */}
    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
      <Text style={styles.monthNumber}>{currentYear}</Text>
      <Text style={styles.monthUnit}>년 </Text>
      <Text style={styles.monthNumber}>{currentMonth + 1}</Text>
      <Text style={styles.monthUnit}>월</Text>
    </View>

    <TouchableOpacity onPress={() => changeMonth('next')}>
      <Text style={styles.arrow}>▶</Text>
    </TouchableOpacity>
  </View>
</View>


      {/* 캘린더 */}
      <View style={styles.calendarContainer}>
        <View style={styles.weekHeader}>
          {['일', '월', '화', '수', '목', '금', '토'].map((d) => (
            <View key={d} style={styles.dayHeader}>
              <Text style={styles.dayHeaderText}>{d}</Text>
            </View>
          ))}
        </View>

        <View style={styles.calendarGrid}>
          {calendarDays.map((day, idx) =>
            day ? (
              <TouchableOpacity
                key={day}
                style={[
                  styles.calendarDay,
                  day === todayDate &&
                    currentMonth === todayMonth &&
                    currentYear === todayYear &&
                    styles.todayStyle,
                ]}
                onPress={() => handleDatePress(day)}
              >
                <Text style={styles.dayText}>{day}</Text>
              </TouchableOpacity>
            ) : (
              <View key={idx} style={styles.calendarDay} />
            )
          )}
        </View>
      </View>

      {/* 자투리 리스트 */}
      <View style={styles.bookReviewBanner}>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <Text style={{ fontSize: 18 }}>📚</Text>
          <Text style={styles.bannerTitle}>  자투리 리스트</Text>
        </View>

        <TouchableOpacity
          onPress={() => router.push('../book-reviews')}
          style={styles.bannerButton}
        >
          <Text style={styles.bannerText}>보기</Text>
        </TouchableOpacity>
      </View>

      {/* 모달 */}
      <Modal transparent visible={showDateModal}>
        <View style={globalStyles.modalOverlay}>
          <View style={globalStyles.modalContent}>
            <Text style={globalStyles.modalTitle}>
              {selectedCalendarDate?.displayDate}
            </Text>

            <Text style={globalStyles.modalText}>
              새로운 글을 작성하시겠습니까?
            </Text>

            <View style={globalStyles.modalButtons}>
              <TouchableOpacity
                style={[
                  globalStyles.button,
                  globalStyles.primaryButton,
                ]}
                onPress={() => handleDateAction('write')}
              >
                <Text style={globalStyles.buttonText}>작성</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[
                  globalStyles.button,
                  globalStyles.secondaryButton,
                ]}
                onPress={() => setShowDateModal(false)}
              >
                <Text style={globalStyles.secondaryButtonText}>취소</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  monthNavigation: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: 220,
    marginTop: 10,
  },
  arrow: {
    fontSize: 20,
    padding: 10,
    color: colors.primary,
  },
  monthNumber: {
    fontFamily: 'TitleFont',
    fontSize: 18,
    fontWeight: 'bold',
    color: colors.dark,
  },
  monthUnit: {
    fontSize: 18,
    color: colors.dark,
  },
  calendarContainer: {
    backgroundColor: colors.white,
    padding: 15,
    borderRadius: 12,
    marginBottom: 20,
  },
  weekHeader: {
    flexDirection: 'row',
  },
  dayHeader: {
    width: (screenWidth - 80) / 7,
    alignItems: 'center',
  },
  dayHeaderText: {
    fontFamily: 'SubTitleFont',
    fontSize: 14,
    color: colors.gray,
  },
  calendarGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  calendarDay: {
    width: (screenWidth - 80) / 7,
    height: 50,
    justifyContent: 'center',
    alignItems: 'center',
  },
  dayText: {
    fontFamily: 'DefaultFont',
    fontSize: 14,
    color: colors.dark,
  },
  todayStyle: {
    backgroundColor: colors.primary,
    borderRadius: 6,
  },
  bookReviewBanner: {
    backgroundColor: colors.primary,
    padding: 15,
    borderRadius: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  bannerTitle: {
    fontFamily: 'TitleFont',
    fontSize: 18,
    color: colors.white,
  },
  bannerButton: {
    backgroundColor: colors.secondary,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 20,
  },
  bannerText: {
    fontFamily: 'SubTitleFont',
    fontSize: 14,
    color: colors.white,
  },
});
