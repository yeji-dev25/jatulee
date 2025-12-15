import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, ScrollView, Alert, StyleSheet, Image } from 'react-native';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage'; // AsyncStorage import
import { globalStyles, colors } from '../../styles/globalStyles';
import { getUserProfile } from '../../api/services'; // getUserProfile API 호출 함수 import

interface User {
  id: number;
  username: string;
  email: string;
  name: string;
  joinDate: string;
  nickName?: string;
  profileImage?: string | null;
}

export default function MyPageScreen() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);  // 사용자 정보 상태
  const [email, setEmail] = useState(''); 
  const [nickName, setNickName] = useState('');
  const [gender, setGender] = useState('');
  const [profileImage, setProfileImage] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true); // 로딩 상태 추가

  useEffect(() => {
    loadData(); // 데이터 로드 함수 호출
  }, []);

  // 데이터 로딩 함수
  const loadData = async () => {
  try {
    const token = await AsyncStorage.getItem('access_token');

    if (!token) {
      Alert.alert("로그인 정보가 없습니다.");
      return;
    }

    const profileData = await getUserProfile(); // 이제 token, userId 안 넣음

    setUser({
      id: profileData.userId,
      email: profileData.email,
      username: profileData.nickName,
      nickName: profileData.nickName,   // 🔥 추가
      name: profileData.nickName,
      joinDate: "",
      profileImage: profileData.profileURL
    });
  } catch (error) {
    console.error('데이터 로드 실패:', error);
    Alert.alert("오류", "데이터 로드 실패");
  } finally {
    setLoading(false);
  }
};


  // 로그아웃 함수
  const handleLogout = async () => {
    Alert.alert(
      '로그아웃',
      '정말 로그아웃하시겠습니까?',
      [
        { text: '취소', style: 'cancel' },
        { 
          text: '로그아웃', 
          onPress: async () => {
            try {
              await AsyncStorage.removeItem('token'); // 토큰 삭제
              await AsyncStorage.removeItem('user'); // 사용자 정보 삭제
              router.replace('/' as any); // 로그인 화면으로 이동
            } catch (error) {
              console.error('로그아웃 실패:', error);
            }
          },
          style: 'destructive' 
        }
      ]
    );
  };

  // 로딩 상태 처리
  if (loading) {
    return <Text>로딩 중...</Text>; // 로딩 중 텍스트
  }

  // 메뉴 항목 리스트
  const menuItems = [
    {
      icon: '✏️',
      title: '프로필 편집',
      onPress: () => router.push('editprofile' as any)
    },
    {
      icon: '📝',
      title: '활동 기록',
      onPress: () => router.push('/(tabs)/calendar' as any)
    },
    {
      icon: '👥',
      title: '친구 관리',
      onPress: () => router.push('/friends' as any)
    }
  ];

  return (
    <ScrollView style={globalStyles.screen}>
      {/* 헤더 */}
      <View style={globalStyles.header}>
        <Text style={globalStyles.title}>마이페이지</Text>
      </View>

      {/* 프로필 카드 */}
      <View style={styles.profileCard}>
        <View style={styles.profileHeader}>
          <View style={styles.avatarContainer}>
            {user?.profileImage ? (
              <Image
                source={{ uri: user.profileImage }}
                style={styles.avatarImage}
              />
            ) : (
              <Text style={styles.avatar}>👤</Text>
            )}
          </View>
          <View style={styles.profileInfo}>
            {/* 사용자 이름 표시 */}
           <Text style={styles.profileName}>
             {user?.email || '이메일이 없습니다'}
              </Text>

            <Text style={styles.profileBio}>
             {user?.nickName || '닉네임이 없습니다'}
                </Text>
          </View>
        </View>
      </View>

      {/* 메뉴 리스트 */}
      <View style={styles.menuSection}>
        {menuItems.map((item, index) => (
          <TouchableOpacity 
            key={index}
            style={styles.menuItem}
            onPress={item.onPress}
          >
            <Text style={styles.menuIcon}>{item.icon}</Text>
            <Text style={styles.menuText}>{item.title}</Text>
            <Text style={styles.menuArrow}></Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* 하단 안내 */}
      <View style={styles.bottomInfo}>
        <Text style={styles.infoText}>버전: 1.0.0</Text>
      </View>

      {/* 로그아웃 버튼 */}
      <View style={styles.logoutSection}>
        <TouchableOpacity 
          style={[globalStyles.button, globalStyles.dangerButton]}
          onPress={handleLogout}
        >
          <Text style={globalStyles.buttonText}>로그아웃</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  profileCard: {
    backgroundColor: colors.white,
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  profileHeader: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
  },
  avatarContainer: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: colors.light,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    marginRight: 15,
    overflow: 'hidden',
  },
  avatarImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  avatar: {
    fontSize: 30,
  },
  profileInfo: {
    flex: 1,
  },
  profileName: {
    fontSize: 20,
    fontWeight: 'bold' as const,
    color: colors.dark,
    marginBottom: 5,
  },
  profileBio: {
    fontSize: 14,
    color: colors.gray,
    marginBottom: 10,
  },
  menuSection: {
    backgroundColor: colors.white,
    borderRadius: 12,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  menuItem: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: colors.light,
  },
  menuIcon: {
    fontSize: 20,
    marginRight: 15,
    width: 25,
  },
  menuText: {
    flex: 1,
    fontSize: 16,
    color: colors.dark,
  },
  menuArrow: {
    fontSize: 16,
    color: colors.gray,
  },
  bottomInfo: {
    backgroundColor: colors.light,
    padding: 15,
    borderRadius: 8,
    marginBottom: 20,
  },
  infoText: {
    fontSize: 12,
    color: colors.gray,
    marginBottom: 5,
  },
  logoutSection: {
    marginBottom: 30,
  },
});
