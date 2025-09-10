"use client";

import React from "react";
import { NavBar } from "./NavBar";
import { usePathname } from "next/navigation";

/**
 * 조건부로 NavBar를 렌더링하는 컴포넌트
 * 
 * 초기페이지(/)에서는 NavBar를 숨기고, 다른 페이지에서는 표시합니다.
 */
const ConditionalNavBar: React.FC = () => {
  const pathname = usePathname();

  // 초기페이지(/)에서는 NavBar 숨기기
  if (pathname === '/') {
    return null;
  }

  // 다른 페이지에서는 NavBar 표시
  return <NavBar animateOnMount={false} />;
};

export default ConditionalNavBar; 