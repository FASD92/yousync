"use client";
import { GoogleLoginButton } from "react-social-login-buttons";
import { GoogleAuthProvider, getIdToken } from "firebase/auth";
// import { auth } from "../../Firebase/firebase-config";
import axios from "axios";
import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { FaSignInAlt } from "react-icons/fa";
import { useRouter } from "next/navigation";
import { auth, provider, signInWithPopup } from "@/lib/firebaseAuth";

const Login2Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 80vh;
  overflow: hidden;
`;

const TextContainer = styled.div`
  font-size: 1.5rem;
  margin-top: 10vh;
  margin-bottom: 15vh;
  line-height: 3rem;
  display:flex;
  justify-content: center;
  align-items: center;

    @media screen and (min-width: 1280px) {
            margin-bottom: 15vh;
        }
`;

const StyledGoogleLoginButton = styled(GoogleLoginButton)`
  width: 70vw !important;
  font-size:17px !important;
  display: flex !important;
  margin-left:15vw !important;
  @media all and (min-width: 768px) and (max-width: 3000px) {
    width: 38vw !important;
    margin-left:31vw !important;
    display:flex !important;
    justify-content: center;
  }
`;

const Login2 = () => {
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  async function handleGoogleLogin() {
    setLoading(true);
    const provider = new GoogleAuthProvider();
    try {
      const result = await signInWithPopup(auth, provider);
      const token = await getIdToken(result.user);
      localStorage.setItem('authToken', token);

      // 서버에 토큰을 보내 회원가입 여부 확인
      const response = await axios.get("NEXT_PUBLIC_API_BASE_UR/auth/google-login", {
        headers: {
          Authorization: `Bearer ${token}`,
          'Cache-Control': 'no-cache'
        }
      });

      if (response.data === "처음 회원 가입한 사용자") {
        router.push('/login3');
      } 
      else if (response.data === "이미 회원 가입된 사용자") {
        router.push('/home');
      }

    } catch (error) { 
      console.error(error);
    } 
  }

  return (
    <Login2Container>
      <TextContainer>
        <p>
          <strong>몸</strong> 과 <strong>마음</strong><br />
          건강하게 챙기는 첫 단계 !
        </p>
      </TextContainer>
      <button
        onClick={() => {
          alert('클릭됨');
          router.push('/login');
        }}
        className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
      >
        <FaSignInAlt />
        로그인
      </button>
    </Login2Container>
  );
};

export default Login2;