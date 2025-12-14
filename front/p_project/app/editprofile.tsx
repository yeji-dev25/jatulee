import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  Alert,
  StyleSheet,
  Image,
  ActivityIndicator,
} from "react-native";
import { useRouter } from "expo-router";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as ImagePicker from "expo-image-picker";
import { uploadApi } from "../api/upladApi";

import {
  updateUserProfile,
  getMyPage,
  updateProfileImage,
} from "../api/services";
import { globalStyles, colors } from "../styles/globalStyles";

interface User {
  id: number;
  username: string;
  email: string;
  gender: string | null;
  profileImage?: string | null;
}

export default function ProfileEditScreen() {
  const router = useRouter();

  const [user, setUser] = useState<User | null>(null);
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [gender, setGender] = useState("");

  const [profileImage, setProfileImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploadingImage, setUploadingImage] = useState(false);
  

  useEffect(() => {
    loadUserData();
  }, []);

  // =========================
  // 🔹 GET /api/mypage
  // =========================
  const loadUserData = async () => {
    try {
      const data = await getMyPage();

      const userData: User = {
        id: data.userId,
        username: data.nickName,
        email: data.email,
        gender: data.gender ?? "",
        profileImage: data.profileURL ?? null,
      };

      setUser(userData);
      setUsername(userData.username);
      setEmail(userData.email);
      setGender(userData.gender || "");
      setProfileImage(userData.profileImage ?? null);

      await AsyncStorage.setItem("user", JSON.stringify(userData));
    } catch (error) {
      console.error("사용자 데이터 불러오기 실패:", error);
      Alert.alert("오류", "프로필 정보를 불러올 수 없습니다.");
    }
  };

  // =========================
  // 🔹 프로필 이미지 선택
  // =========================
  const pickProfileImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (result.canceled) return;

    const asset = result.assets[0];
    await uploadProfileImage(asset);
  };

  // =========================
  // 🔹 POST /api/mypage/profile
  // =========================
  const uploadProfileImage = async (
    asset: ImagePicker.ImagePickerAsset
  ) => {
    try {
      setUploadingImage(true);

      const res = await updateProfileImage({
        uri: asset.uri,
        name: "profile.jpg",
        type: "image/jpeg",
      });

      setProfileImage(res.profileURL);

      if (user) {
        const updatedUser = {
          ...user,
          profileImage: res.profileURL,
        };
        setUser(updatedUser);
        await AsyncStorage.setItem("user", JSON.stringify(updatedUser));
      }
    } catch (error) {
      console.error("프로필 이미지 업로드 실패:", error);
      Alert.alert("오류", "프로필 이미지 업로드에 실패했습니다.");
    } finally {
      setUploadingImage(false);
    }
  };

  // =========================
  // 🔹 POST /api/mypage/update
  // =========================
  const handleSave = async () => {
    if (!username.trim() || !email.trim()) {
      Alert.alert("알림", "닉네임과 이메일은 반드시 필요합니다.");
      return;
    }

    if (!user?.id) {
      Alert.alert("오류", "사용자 정보를 찾을 수 없습니다.");
      return;
    }

    setLoading(true);

    try {
      const updatedUser = await updateUserProfile(
        user.id,
        email,
        username,
        gender
      );

      await AsyncStorage.setItem("user", JSON.stringify(updatedUser));
      setUser(updatedUser);

      Alert.alert("성공", "프로필이 업데이트되었습니다.");
      router.back();
    } catch (error: any) {
      console.error("프로필 업데이트 실패:", error);

      if (error?.response?.status === 409) {
        Alert.alert("오류", "이미 존재하는 이메일 또는 닉네임입니다.");
      } else {
        Alert.alert("오류", "프로필 업데이트 중 문제가 발생했습니다.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={globalStyles.screen}>
      <View style={globalStyles.header}>
        <Text style={globalStyles.title}>프로필 편집</Text>
      </View>

      <View style={styles.card}>
        {/* 프로필 이미지 */}
        <View style={styles.profileImageWrapper}>
          <TouchableOpacity onPress={pickProfileImage}>
            <Image
              source={
                profileImage
                  ? { uri: profileImage }
                  : require("../assets/images/icon.png")


              }
              style={styles.profileImage}
            />

            {uploadingImage && (
              <View style={styles.imageOverlay}>
                <ActivityIndicator color={colors.white} />
              </View>
            )}
          </TouchableOpacity>
          <Text style={styles.changeText}>프로필 사진 변경</Text>
        </View>

        <TextInput
          style={styles.textInput}
          value={username}
          onChangeText={setUsername}
          placeholder="닉네임"
        />

        <TextInput
          style={styles.textInput}
          value={email}
          onChangeText={setEmail}
          placeholder="이메일"
          keyboardType="email-address"
        />

        <TextInput
          style={styles.textInput}
          value={gender}
          onChangeText={setGender}
          placeholder="성별"
        />

        <TouchableOpacity
          style={styles.button}
          onPress={handleSave}
          disabled={loading}
        >
          <Text style={styles.buttonText}>
            {loading ? "저장 중..." : "저장"}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.white,
    padding: 25,
    borderRadius: 12,
    marginTop: 20,
    marginHorizontal: 20,
    elevation: 6,
  },
  profileImageWrapper: {
    alignItems: "center",
    marginBottom: 25,
  },
  profileImage: {
    width: 110,
    height: 110,
    borderRadius: 55,
    backgroundColor: colors.light,
  },
  imageOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    borderRadius: 55,
    backgroundColor: "rgba(0,0,0,0.4)",
    justifyContent: "center",
    alignItems: "center",
  },
  changeText: {
    marginTop: 10,
    fontSize: 14,
    color: colors.primary,
    fontWeight: "500",
  },
  textInput: {
    backgroundColor: colors.light,
    borderRadius: 8,
    paddingVertical: 14,
    paddingHorizontal: 18,
    fontSize: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: colors.lightGray,
  },
  button: {
    backgroundColor: colors.primary,
    paddingVertical: 16,
    borderRadius: 8,
    alignItems: "center",
  },
  buttonText: {
    fontSize: 18,
    color: colors.white,
    fontWeight: "600",
  },
});
