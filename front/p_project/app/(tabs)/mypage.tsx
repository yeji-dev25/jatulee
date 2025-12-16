import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, ScrollView, Alert, StyleSheet, Image } from 'react-native';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { globalStyles, colors } from '../../styles/globalStyles';
import { getUserProfile } from '../../api/services';

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
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const token = await AsyncStorage.getItem('access_token');
      if (!token) {
        Alert.alert('로그인 정보가 없습니다.');
        return;
      }

      const profileData = await getUserProfile();

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

  const handleLogout = async () => {
    Alert.alert('로그아웃', '정말 로그아웃하시겠습니까?', [
      { text: '취소', style: 'cancel' },
      {
        text: '로그아웃',
        style: 'destructive',
        onPress: async () => {
          await AsyncStorage.removeItem('token');
          await AsyncStorage.removeItem('user');
          router.replace('/' as any);
        },
      },
    ]);
  };

  if (loading) {
    return (
      <Text style={{ fontFamily: 'DefaultFont', textAlign: 'center', marginTop: 50 }}>
        로딩 중...
      </Text>
    );
  }

  const menuItems = [
    { icon: '✏️', title: '프로필 편집', onPress: () => router.push('editprofile' as any) },
    { icon: '📝', title: '활동 기록', onPress: () => router.push('/(tabs)/calendar' as any) },
    { icon: '👥', title: '친구 관리', onPress: () => router.push('/friends' as any) },
  ];

  return (
    <ScrollView style={globalStyles.screen}>
      {/* 헤더 */}
      <View style={globalStyles.header}>
        <Text
  style={{
    fontFamily: 'SubTitleFont', // 또는 TitleFont
    fontSize: 24,
    color: colors.dark,
    marginBottom: 5,
  }}
>
  마이페이지
</Text>
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
            <Text style={styles.profileName}>
              {user?.email || '이메일이 없습니다'}
            </Text>
            <Text style={styles.profileBio}>
              {user?.nickName || '닉네임이 없습니다'}
            </Text>
          </View>
        </View>
      </View>

      {/* 메뉴 */}
      <View style={styles.menuSection}>
        {menuItems.map((item, index) => (
          <TouchableOpacity key={index} style={styles.menuItem} onPress={item.onPress}>
            <Text style={styles.menuIcon}>{item.icon}</Text>
            <Text style={styles.menuText}>{item.title}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* 하단 정보 */}
      <View style={styles.bottomInfo}>
        <Text style={styles.infoText}>버전: 1.0.0</Text>
      </View>

      {/* 로그아웃 */}
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
    elevation: 3,
  },
  profileHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  avatarContainer: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: colors.light,
    alignItems: 'center',
    justifyContent: 'center',
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
    fontFamily: 'TitleFont',
    fontSize: 20,
    color: colors.dark,
    marginBottom: 5,
  },
  profileBio: {
    fontFamily: 'SubTitleFont',
    fontSize: 14,
    color: colors.gray,
  },

  menuSection: {
    backgroundColor: colors.white,
    borderRadius: 12,
    marginBottom: 20,
    elevation: 3,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
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
    fontFamily: 'DefaultFont',
    fontSize: 16,
    color: colors.dark,
  },

  bottomInfo: {
    backgroundColor: colors.light,
    padding: 15,
    borderRadius: 8,
    marginBottom: 20,
  },
  infoText: {
    fontFamily: 'DefaultFont',
    fontSize: 12,
    color: colors.gray,
  },

  logoutSection: {
    marginBottom: 30,
  },
});
