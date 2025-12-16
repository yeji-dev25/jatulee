import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Modal,
  Alert,
  StyleSheet,
} from 'react-native';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { globalStyles, colors } from '../../styles/globalStyles';
import {
  getFriendRequests,
  sendFriendRequest,
  acceptFriendRequest,
  rejectFriendRequest,
  getFriendList,
} from '../../api/services';

interface Friend {
  id: number;
  nickname: string;
  email: string;
}

export default function FriendsScreen() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'list' | 'requests'>('list');
  const [friends, setFriends] = useState<Friend[]>([]);
  const [friendRequests, setFriendRequests] = useState<Friend[]>([]);
  const [searchText, setSearchText] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);
  const [newFriendEmail, setNewFriendEmail] = useState('');

  useEffect(() => {
    setTimeout(loadData, 300);
  }, []);

  const loadData = async () => {
    try {
      const requests = await getFriendRequests();
      const list = await getFriendList();
      setFriendRequests(requests);
      setFriends(list);
    } catch (e) {
      Alert.alert('오류', '친구 정보를 불러오지 못했습니다.');
    }
  };

  const addFriend = async () => {
    if (!newFriendEmail.trim()) {
      return Alert.alert('알림', '친구 이메일을 입력해주세요.');
    }
    await sendFriendRequest(newFriendEmail);
    setNewFriendEmail('');
    setShowAddModal(false);
    loadData();
  };

  /* =========================
     Friend List
  ========================= */
  const FriendListTab = () => (
    <View>
      <View style={styles.friendActions}>
        <TextInput
          style={styles.searchInput}
          placeholder="친구 검색..."
          placeholderTextColor={colors.gray}
          value={searchText}
          onChangeText={setSearchText}
        />

        <TouchableOpacity
          onPress={() => setShowAddModal(true)}
          style={styles.addButton}
        >
          <Text style={styles.addButtonText}>+ 추가</Text>
        </TouchableOpacity>
      </View>

      <ScrollView>
        {friends.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyIcon}>👥</Text>
            <Text style={styles.emptyText}>친구가 없습니다.</Text>
          </View>
        ) : (
          friends.map(friend => (
            <View key={friend.id} style={styles.friendCard}>
              <TouchableOpacity
                onPress={() =>
                  router.push({
                    pathname: '/friends/friendCalendar',
                    params: { friendId: friend.id.toString() },
                  })
                }
              >
                <Text style={styles.friendName}>
                  {friend.nickname}
                </Text>
              </TouchableOpacity>
            </View>
          ))
        )}
      </ScrollView>
    </View>
  );

  /* =========================
     Friend Requests
  ========================= */
  const FriendRequestsTab = () => (
    <ScrollView>
      {friendRequests.length === 0 ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyIcon}>📬</Text>
          <Text style={styles.emptyText}>새로운 요청이 없습니다.</Text>
        </View>
      ) : (
        friendRequests.map(req => (
          <View key={req.id} style={styles.requestCard}>
            <View>
              <Text style={styles.requestName}>{req.nickname}</Text>
              <Text style={styles.requestSub}>친구 요청</Text>
            </View>

            <View style={styles.requestButtons}>
              <TouchableOpacity
                style={styles.acceptButton}
                onPress={() => acceptFriendRequest(req.id)}
              >
                <Text style={styles.acceptButtonText}>✔ 수락</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.rejectButton}
                onPress={() => rejectFriendRequest(req.id)}
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
                  친구
                 </Text>
      </View>

      {/* 탭 */}
      <View style={styles.tabHeader}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'list' && styles.activeTab]}
          onPress={() => setActiveTab('list')}
        >
          <Text style={styles.tabText}>친구 목록</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.tab, activeTab === 'requests' && styles.activeTab]}
          onPress={() => setActiveTab('requests')}
        >
          <Text style={styles.tabText}>친구 요청</Text>
        </TouchableOpacity>
      </View>

      {activeTab === 'list' ? <FriendListTab /> : <FriendRequestsTab />}

      {/* 친구 추가 모달 */}
      <Modal visible={showAddModal} transparent animationType="fade">
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
            />

            <TouchableOpacity
              style={styles.modalPrimaryButton}
              onPress={addFriend}
            >
              <Text style={styles.modalPrimaryText}>📨 요청 보내기</Text>
            </TouchableOpacity>

            <TouchableOpacity onPress={() => setShowAddModal(false)}>
              <Text style={styles.modalCancelText}>취소</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

/* =========================
   Styles
========================= */
const styles = StyleSheet.create({
  pageTitle: {
    fontFamily: 'SubTitleFont',
    fontSize: 24,
    fontWeight: '700',
    color: colors.dark,
  },

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
  tabText: {
    fontFamily: 'DefaultFont',
    fontSize: 15,
    color: colors.dark,
  },

  friendActions: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 15,
  },
  searchInput: {
    flex: 1,
    backgroundColor: colors.lightGray,
    borderRadius: 20,
    paddingHorizontal: 14,
    fontFamily: 'DefaultFont',
    fontSize: 16,
  },

  addButton: {
    backgroundColor: colors.primary,
    paddingHorizontal: 16,
    borderRadius: 25,
    justifyContent: 'center',
  },
  addButtonText: {
    fontFamily: 'SubTitleFont',
    fontSize: 16,
    color: colors.white,
  },

  emptyState: {
    alignItems: 'center',
    marginTop: 50,
  },
  emptyIcon: {
    fontSize: 48,
    marginBottom: 16,
  },
  emptyText: {
    fontFamily: 'DefaultFont',
    fontSize: 14,
    color: colors.gray,
  },

  friendCard: {
    backgroundColor: colors.white,
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
    elevation: 3,
  },
  friendName: {
    fontFamily: 'TitleFont',
    fontSize: 16,
    color: colors.dark,
  },

  requestCard: {
    backgroundColor: colors.white,
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  requestName: {
    fontFamily: 'TitleFont',
    fontSize: 16,
  },
  requestSub: {
    fontFamily: 'DefaultFont',
    fontSize: 12,
    color: colors.gray,
  },

  requestButtons: {
    flexDirection: 'row',
    gap: 8,
  },
  acceptButton: {
    backgroundColor: colors.primary,
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 6,
  },
  acceptButtonText: {
    fontFamily: 'SubTitleFont',
    color: colors.white,
  },
  rejectButton: {
    backgroundColor: colors.danger,
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 6,
  },
  rejectButtonText: {
    fontFamily: 'SubTitleFont',
    color: colors.white,
  },

  modalOverlay: {
    flex: 1,
    backgroundColor: '#00000066',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: '80%',
    backgroundColor: colors.white,
    borderRadius: 14,
    padding: 20,
    alignItems: 'center',
  },
  modalTitle: {
    fontFamily: 'TitleFont',
    fontSize: 20,
    marginBottom: 6,
  },
  modalDesc: {
    fontFamily: 'DefaultFont',
    fontSize: 13,
    color: colors.gray,
    marginBottom: 20,
    textAlign: 'center',
  },
  modalInput: {
    width: '100%',
    backgroundColor: colors.lightGray,
    borderRadius: 20,
    paddingHorizontal: 14,
    fontFamily: 'DefaultFont',
    marginBottom: 20,
  },
  modalPrimaryButton: {
    backgroundColor: colors.primary,
    borderRadius: 14,
    paddingVertical: 14,
    width: '100%',
    alignItems: 'center',
    marginBottom: 12,
  },
  modalPrimaryText: {
    fontFamily: 'SubTitleFont',
    fontSize: 16,
    color: colors.white,
  },
  modalCancelText: {
    fontFamily: 'DefaultFont',
    fontSize: 14,
    color: colors.gray,
  },
});
