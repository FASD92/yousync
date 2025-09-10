import { getAuth, GoogleAuthProvider, signInWithPopup } from "firebase/auth";
import { initializeApp } from "firebase/app";

// yousync-a8fcc firebaseConfig
const firebaseConfig = {
  apiKey: "AIzaSyBYeNY_pnH4EO-02b2XCGUNan7NYPGt4OY", // 실제 프로젝트의 값 입력
  authDomain: "yousync-a8fcc.firebaseapp.com",
  projectId: "yousync-a8fcc",
  storageBucket: "yousync-a8fcc.appspot.com",
  messagingSenderId: "708460483943",
  appId: "1:708460483943:web:YOUR_REAL_APP_ID",         // 실제 값으로 교체
  measurementId: "G-REPLACE_ME"                        // 실제 값으로 교체
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

export { auth, provider, signInWithPopup };
