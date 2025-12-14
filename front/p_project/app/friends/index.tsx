import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, Modal, Alert, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { globalStyles, colors } from '../../styles/globalStyles';
import {
  getFriendRequests,
  sendFriendRequest,
  acceptFriendRequest,
  rejectFriendRequest,
  getFriendList,
  removeFriend
} from '../../api/services';

interface Friend {
  id: number;
  nickname: string;
  email: string;
}

export default function FriendsScreen() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState('list');
  const [friends, setFriends] = useState<Friend[]>([]);
  const [friendRequests, setFriendRequests] = useState<Friend[]>([]);
  const [searchText, setSearchText] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);
  const [newFriendEmail, setNewFriendEmail] = useState('');

  const getUserData = async () => {
    const accessToken = await AsyncStorage.getItem('access_token');
    const userIdStr = await AsyncStorage.getItem('user_id');

    if (!accessToken || !userIdStr) {
      return { token: null, userId: null };
    }

    return { token: accessToken, userId: Number(userIdStr) };
  };

  // 데이터 로드
  useEffect(() => {
    setTimeout(() => {
      loadData();
    }, 300); // 300ms 지연
  }, []);

  const loadData = async () => {
    try {
      const { token, userId } = await getUserData();
      if (!token || !userId) return Alert.alert('알림', '로그인 정보 없음');

      const requestData = await getFriendRequests();
      setFriendRequests(requestData);

      const friendsData = await getFriendList();
      setFriends(friendsData);

    } catch (error) {
      console.error("친구 데이터 로드 실패:", error);
      Alert.alert("오류", "친구 정보를 불러오지 못했습니다.");
    }
  };

  // 친구 요청 보내기
  const addFriend = async () => {
    if (!newFriendEmail.trim()) {
      return Alert.alert('알림', '친구 이메일을 입력해주세요.');
    }

    try {
      const { token, userId } = await getUserData();
      if (!token || !userId) return;

      await sendFriendRequest(newFriendEmail);
      Alert.alert('완료', '친구 요청을 보냈습니다.');
      setNewFriendEmail('');
      setShowAddModal(false);
      loadData();
    } catch (error) {
      console.error('친구 요청 보내기 실패:', error);
      Alert.alert('오류', '친구 요청 보내기 실패');
    }
  };

  // 친구 요청 수락
  const acceptRequestHandler = async (fromUserId: number) => {
    try {
      const { token, userId } = await getUserData();
      if (!token || !userId) return;

      await acceptFriendRequest(fromUserId);
      Alert.alert('완료', '친구 요청을 수락했습니다!');
      loadData();
    } catch (error) {
      console.error('친구 요청 수락 실패:', error);
      Alert.alert('오류', '친구 요청 수락 실패');
    }
  };

  // 친구 요청 거절
  const rejectRequestHandler = async (fromUserId: number) => {
    try {
      const { token, userId } = await getUserData();
      if (!token || !userId) return;

      await rejectFriendRequest(fromUserId);
      Alert.alert('완료', '친구 요청을 거절했습니다.');
      loadData();
    } catch (error) {
      console.error('친구 요청 거절 실패:', error);
      Alert.alert('오류', '친구 요청 거절 실패');
    }
  };

  // 친구 목록 UI
  const FriendListTab = () => (
    <View>
      <View style={styles.friendActions}>
        <TextInput
          style={globalStyles.searchInput}
          placeholder="친구 검색..."
          value={searchText}
          onChangeText={setSearchText}
        />
        <TouchableOpacity onPress={() => setShowAddModal(true)} style={styles.addButton}>
          <Text style={styles.addButtonText}>+ 추가</Text>
        </TouchableOpacity>
      </View>

      <ScrollView>
        {friends.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyIcon}>👥</Text>
            <Text style={globalStyles.emptyText}>친구가 없습니다.</Text>
          </View>
        ) : (
          friends.map(friend => (
            <View key={friend.id} style={styles.friendCard}>
              {/* 친구 닉네임 클릭 → 친구 캘린더로 이동 */}
              <TouchableOpacity
                onPress={() =>
                  router.push({
                    pathname: "/friends/friendCalendar",
                    params: { friendId: friend.id.toString() } // friend.id 대신 userId로 수정 필요
                  })
                }
              >
                <Text>{friend.nickname}</Text>
              </TouchableOpacity>
            </View>
          ))
        )}
      </ScrollView>
    </View>
  );

  // 친구 요청 UI
 const FriendRequestsTab = () => (
  <ScrollView>
    {friendRequests.length === 0 ? (
      <View style={styles.emptyState}>
        <Text style={styles.emptyIcon}>📬</Text>
        <Text style={globalStyles.emptyText}>새로운 요청이 없습니다.</Text>
      </View>
    ) : (
      friendRequests.map(request => (
        <View key={request.id} style={styles.requestCard}>
          
          {/* 닉네임 */}
          <View>
            <Text style={styles.requestName}>{request.nickname}</Text>
            <Text style={styles.requestSub}>친구 요청</Text>
          </View>

          {/* 버튼 그룹 */}
          <View style={styles.requestButtons}>
            <TouchableOpacity
              style={styles.acceptButton}
              onPress={() => acceptRequestHandler(request.id)}
              activeOpacity={0.8}
            >
              <Text style={styles.acceptButtonText}>✔ 수락</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.rejectButton}
              onPress={() => rejectRequestHandler(request.id)}
              activeOpacity={0.8}
            >
              <Text style={styles.rejectButtonText}>✖ 거절</Text>
            </TouchableOpacity>
          </View>

        </View>
      ))
    )}
  </ScrollView>
);
  return (
    <View style={globalStyles.screen}>
      <View style={globalStyles.header}>
        <Text style={globalStyles.title}>친구</Text>
      </View>

      <View style={styles.tabHeader}>
        <TouchableOpacity
          onPress={() => setActiveTab('list')}
          style={[styles.tab, activeTab === 'list' && styles.activeTab]}
        >
          <Text>친구 목록</Text>
        </TouchableOpacity>
        <TouchableOpacity
          onPress={() => setActiveTab('requests')}
          style={[styles.tab, activeTab === 'requests' && styles.activeTab]}
        >
          <Text>친구 요청</Text>
        </TouchableOpacity>
      </View>

      {activeTab === 'list' ? <FriendListTab /> : <FriendRequestsTab />}

      {/* 친구 추가 모달 */}
     <Modal visible={showAddModal} animationType="fade" transparent>
  <View style={styles.modalOverlay}>
    <View style={styles.modalContent}>

      <Text style={styles.modalTitle}>친구 추가</Text>
      <Text style={styles.modalDesc}>
        친구의 이메일을 입력하면 요청이 전송됩니다.
      </Text>

      <TextInput
        style={styles.modalInput}
        placeholder="example@email.com"
        placeholderTextColor={colors.gray}
        value={newFriendEmail}
        onChangeText={setNewFriendEmail}
        keyboardType="email-address"
        autoCapitalize="none"
      />

      <TouchableOpacity
        style={styles.modalPrimaryButton}
        onPress={addFriend}
        activeOpacity={0.85}
      >
        <Text style={styles.modalPrimaryText}>📨 요청 보내기</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.modalCancelButton}
        onPress={() => setShowAddModal(false)}
      >
        <Text style={styles.modalCancelText}>취소</Text>
      </TouchableOpacity>

    </View>
  </View>
</Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  friendActions: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 15,
  },
  searchInput: {
    flex: 1,
    padding: 10,
    borderRadius: 20,
    backgroundColor: colors.lightGray,
    fontSize: 16,
  },
  addButton: {
    backgroundColor: colors.primary,
    paddingVertical: 12,
    paddingHorizontal: 15,
    borderRadius: 25,
  },
  addButtonText: { color: colors.white, fontSize: 16, fontWeight: 'bold' },
  emptyState: { alignItems: 'center', marginTop: 50 },
  emptyIcon: { fontSize: 50, marginBottom: 20 },
  friendCard: {
    backgroundColor: colors.white,
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 4,
  },
  friendName: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  removeButton: {
    backgroundColor: colors.danger,
    paddingVertical: 8,
    paddingHorizontal: 15,
    borderRadius: 20,
  },
  removeButtonText: { color: colors.white, fontSize: 14 },
  tabHeader: {
    flexDirection: 'row',
    marginBottom: 20,
  },
  tab: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
    backgroundColor: colors.light,
    borderRadius: 20,
  },
  activeTab: {
    backgroundColor: colors.primary,
  },
  requestCard: {
    backgroundColor: colors.white,
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 4,
  },
  acceptButton: {
    backgroundColor: colors.primary,
    paddingVertical: 8,
    paddingHorizontal: 15,
    borderRadius: 20,
  },
  acceptButtonText: { color: colors.white },
  rejectButton: {
    backgroundColor: colors.danger,
    paddingVertical: 8,
    paddingHorizontal: 15,
    borderRadius: 20,
  },
  rejectButtonText: { color: colors.white },
  modalOverlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#00000066',
  },
  modalContent: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 12,
    width: '80%',
    alignItems: 'center',
  },
  modalInput: {
    width: '100%',
    padding: 10,
    borderRadius: 20,
    backgroundColor: colors.lightGray,
    marginBottom: 20,
  },
  modalButton: {
    backgroundColor: colors.primary,
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 25,
    marginBottom: 10,
  },
  modalButtonText: {
    color: colors.white,
    fontSize: 16,
    fontWeight: 'bold',
  },
  requestName: {
  fontSize: 16,
  fontWeight: '600',
  color: colors.dark,
},

requestSub: {
  fontSize: 12,
  color: colors.gray,
  marginTop: 4,
},

requestButtons: {
  flexDirection: 'row',
  gap: 8,
},

modalTitle: {
  fontSize: 20,
  fontWeight: '700',
  color: colors.dark,
  marginBottom: 6,
},

modalDesc: {
  fontSize: 13,
  color: colors.gray,
  marginBottom: 20,
},


modalPrimaryButton: {
  backgroundColor: colors.primary,

  paddingVertical: 18,        
  paddingHorizontal: 20,     
  borderRadius: 14,

  alignItems: 'center',
  justifyContent: 'center',

  minHeight: 56,              
  marginBottom: 14,
},

modalPrimaryText: {
  color: colors.white,
  fontSize: 16,
  fontWeight: '600',
},

modalCancelButton: {
  alignItems: 'center',
},

modalCancelText: {
  color: colors.gray,
  fontSize: 14,
},

});