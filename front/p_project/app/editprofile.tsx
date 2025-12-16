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
    } catch {
      Alert.alert("오류", "프로필 정보를 불러올 수 없습니다.");
    }
  };

  const pickProfileImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (result.canceled) return;
    await uploadProfileImage(result.assets[0]);
  };

  const uploadProfileImage = async (asset: ImagePicker.ImagePickerAsset) => {
    try {
      setUploadingImage(true);
      const res = await updateProfileImage({
        uri: asset.uri,
        name: "profile.jpg",
        type: "image/jpeg",
      });
      setProfileImage(res.profileURL);
    } catch {
      Alert.alert("오류", "프로필 이미지 업로드 실패");
    } finally {
      setUploadingImage(false);
    }
  };

  const handleSave = async () => {
    if (!username.trim() || !email.trim()) {
      Alert.alert("알림", "닉네임과 이메일은 필수입니다.");
      return;
    }

    setLoading(true);
    try {
      await updateUserProfile(user!.id, email, username, gender);
      Alert.alert("성공", "프로필이 업데이트되었습니다.");
      router.back();
    } catch {
      Alert.alert("오류", "프로필 업데이트 실패");
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={globalStyles.screen}>
      {/* 🔥 헤더 (대제목은 inline) */}
      <View style={globalStyles.header}>
        <Text
  style={{
    fontFamily: 'SubTitleFont',
    fontSize: 24,
    color: colors.dark,
    marginBottom: 5,
  }}
>
  프로필 편집
</Text>
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
  pageTitle: {
    fontFamily: "SubTitleFont",
    fontSize: 24,
    fontWeight: "700",
    color: colors.dark,
    marginBottom: 5,
  },

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
    inset: 0,
    borderRadius: 55,
    backgroundColor: "rgba(0,0,0,0.4)",
    justifyContent: "center",
    alignItems: "center",
  },
  changeText: {
    fontFamily: "SubTitleFont",
    marginTop: 10,
    fontSize: 14,
    color: colors.primary,
  },

  textInput: {
    fontFamily: "DefaultFont",
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
    fontFamily: "SubTitleFont",
    fontSize: 18,
    color: colors.white,
  },
});
