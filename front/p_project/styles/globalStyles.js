// app/styles/globalStyles.js
import { StyleSheet, Dimensions } from 'react-native';

const { width: screenWidth } = Dimensions.get('window');

export const colors = {
  primary: '#0984e3',
  secondary: '#74b9ff',
  success: '#00b894',
  danger: '#e74c3c',
  warning: '#f39c12',
  info: '#0984e3',
  light: '#f8f9fa',
  dark: '#2d3436',
  gray: '#636e72',
  lightGray: '#ddd',
  white: '#ffffff',
  kakaoYellow: '#FEE500',
  googleBlue: '#34b7f1',
  naverGreen: '#00C300',
   primary: '#B17457',      // 메인 버튼
  secondary: '#B17457',  // 보조 강조
  accent: '#8B5A3C',       // 강한 강조
};

export const globalStyles = StyleSheet.create({
  /* =========================
     Layout / Container
  ========================= */
  container: {
    flex: 1,
    backgroundColor: colors.white,
  },
 screen: {
  flex: 1,
  backgroundColor: colors.background, // #FAF7F0
  padding: 20,
},
  scrollView: {
    flex: 1,
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },

  /* =========================
     Loading
  ========================= */
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.light,
  },
  loadingText: {
    marginTop: 10,
    color: colors.primary,
    fontSize: 16,
    fontFamily: 'DefaultFont',
  },

  /* =========================
     Typography (🔥 핵심)
  ========================= */
  title: {
    fontFamily: 'TitleFont',
    fontSize: 24,
    fontWeight: '700',
    color: colors.dark,
  },
  subtitle: {
    fontFamily: 'SubTitleFont',
    fontSize: 16,
    color: colors.gray,
  },
  text: {
    fontFamily: 'DefaultFont',
    fontSize: 14,
    color: colors.dark,
  },
  emptyText: {
    textAlign: 'center',
    color: colors.lightGray,
    fontSize: 16,
    marginTop: 20,
    fontFamily: 'DefaultFont',
  },

  /* =========================
     Header
  ========================= */
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '100%',
    marginBottom: 5,
  },
  headerButtons: {
    flexDirection: 'row',
    gap: 10,
  },

  /* =========================
     Buttons
  ========================= */
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginVertical: 20,
  },
  button: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    minWidth: 100,
    alignItems: 'center',
  },
  primaryButton: {
    backgroundColor: colors.primary,
  },
  secondaryButton: {
   backgroundColor: colors.secondary,
    borderWidth: 1,
    borderColor: colors.primary,
  },
  dangerButton: {
    backgroundColor: colors.danger,
  },
  successButton: {
    backgroundColor: colors.success,
  },
  warningButton: {
    backgroundColor: colors.warning,
  },
  disabledButton: {
    backgroundColor: colors.gray,
    opacity: 0.6,
  },
  buttonText: {
    color: colors.white,
    fontSize: 16,
    fontWeight: '600',
    fontFamily: 'DefaultFont',
  },
  secondaryButtonText: {
     color: colors.white,
    fontSize: 16,
    fontWeight: '600',
    fontFamily: 'DefaultFont',
  },

  /* =========================
     Inputs
  ========================= */
  inputContainer: {
    width: '100%',
    marginBottom: 20,
  },
  inputLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.dark,
    marginBottom: 8,
    fontFamily: 'SubTitleFont',
  },
  textInput: {
    backgroundColor: colors.white,
    borderWidth: 1,
    borderColor: colors.lightGray,
    borderRadius: 8,
    padding: 15,
    fontSize: 16,
    fontFamily: 'DefaultFont',
  },
  searchInput: {
    backgroundColor: colors.white,
    borderWidth: 1,
    borderColor: colors.lightGray,
    borderRadius: 25,
    paddingHorizontal: 20,
    paddingVertical: 12,
    fontSize: 16,
    flex: 1,
    fontFamily: 'DefaultFont',
  },

  /* =========================
     Login / Social
  ========================= */
  loginContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  linkContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 20,
  },
  linkText: {
    color: colors.primary,
    fontSize: 16,
    fontWeight: '600',
    fontFamily: 'DefaultFont',
  },
  linkSeparator: {
    color: colors.gray,
    marginHorizontal: 15,
  },
  socialLoginContainer: {
    marginTop: 20,
    alignItems: 'center',
    width: '100%',
  },
  socialButton: {
    paddingVertical: 15,
    borderRadius: 8,
    marginTop: 10,
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  socialButtonText: {
    color: colors.white,
    fontSize: 16,
    fontWeight: '600',
    fontFamily: 'DefaultFont',
  },

  /* =========================
     Card / List
  ========================= */
  card: {
    backgroundColor: colors.card,
    padding: 20,
    borderRadius: 12,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  listItem: {
    backgroundColor: colors.white,
    padding: 15,
    borderRadius: 8,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 3,
  },
  listItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 5,
  },
  listItemTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: colors.dark,
    flex: 1,
    fontFamily: 'SubTitleFont',
  },
  listItemSubtitle: {
    fontSize: 14,
    color: colors.gray,
    marginBottom: 5,
    fontFamily: 'DefaultFont',
  },

  /* =========================
     Modal
  ========================= */
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: colors.white,
    padding: 20,
    borderRadius: 12,
    width: screenWidth * 0.8,
    maxWidth: 300,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: colors.dark,
    textAlign: 'center',
    marginBottom: 15,
    fontFamily: 'TitleFont',
  },
  modalText: {
    fontSize: 16,
    color: colors.gray,
    textAlign: 'center',
    marginBottom: 20,
    fontFamily: 'DefaultFont',
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  modalButton: {
    minWidth: 80,
  },
});
