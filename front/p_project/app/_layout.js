import { DefaultTheme, ThemeProvider } from '@react-navigation/native';
import { useFonts } from 'expo-font';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import * as WebBrowser from "expo-web-browser";
import * as SplashScreen from "expo-splash-screen";
import { useEffect } from "react";

WebBrowser.maybeCompleteAuthSession();
SplashScreen.preventAutoHideAsync();

/** 🔥 연한 갈색 배경 커스텀 테마 */
const CustomTheme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    background: '#FAF7F0',   // 전체 배경
    card: '#FAF7F0',         // header / 카드 배경
    text: '#6B6966',         // 메인 텍스트
    border: '#E8E4D9',
    primary: '#B17457',      // CTA 강조색
  },
};

export default function RootLayout() {
  const [loaded] = useFonts({
    TitleFont: require('../assets/fonts/FontOTFBold.otf'),
    SubTitleFont: require('../assets/fonts/FontOTFRegular.otf'),
    DefaultFont: require('../assets/fonts/omyupretty.ttf'),
  });

  useEffect(() => {
    if (loaded) {
      SplashScreen.hideAsync();
    }
  }, [loaded]);

  if (!loaded) return null;

  return (
    <ThemeProvider value={CustomTheme}>
      <Stack
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: '#FAF7F0' },
        }}
      >
        <Stack.Screen name="index" />
        <Stack.Screen name="login" />
        <Stack.Screen name="(tabs)" />
        <Stack.Screen name="+not-found" />
      </Stack>

      {/* 상태바도 다크 텍스트로 */}
      <StatusBar style="dark" />
    </ThemeProvider>
  );
}
