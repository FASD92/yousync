"use client";

import { useState, useEffect } from 'react';

interface UseSessionStorageOptions {
  key: string;
  defaultValue?: string;
}

interface UseSessionStorageReturn {
  value: string | null;
  setValue: (value: string) => void;
  removeValue: () => void;
  isInitialized: boolean;
}

/**
 * 세션스토리지 방문 체크를 위한 커스텀 훅
 * 
 * @param options - 세션스토리지 옵션
 * @param options.key - 세션스토리지 키
 * @param options.defaultValue - 기본값 (선택사항)
 * 
 * @returns 세션스토리지 값과 관련 함수들
 * 
 * @example
 * ```tsx
 * const { value, setValue, removeValue, isInitialized } = useSessionStorage({
 *   key: 'hasVisited',
 *   defaultValue: 'false'
 * });
 * 
 * if (!isInitialized) {
 *   return <LoadingSpinner />;
 * }
 * 
 * return value === 'true' ? <HomePage /> : <LetsRunPage />;
 * ```
 */
export function useSessionStorage({ key, defaultValue }: UseSessionStorageOptions): UseSessionStorageReturn {
  const [value, setValueState] = useState<string | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    try {
      // 세션스토리지에서 값 가져오기
      const storedValue = sessionStorage.getItem(key);
      
      if (storedValue !== null) {
        setValueState(storedValue);
      } else if (defaultValue !== undefined) {
        // 기본값이 있고 저장된 값이 없으면 기본값 설정
        sessionStorage.setItem(key, defaultValue);
        setValueState(defaultValue);
      } else {
        setValueState(null);
      }
    } catch (error) {
      console.error('세션스토리지 접근 오류:', error);
      setValueState(defaultValue || null);
    } finally {
      setIsInitialized(true);
    }
  }, [key, defaultValue]);

  const setValue = (newValue: string) => {
    try {
      sessionStorage.setItem(key, newValue);
      setValueState(newValue);
    } catch (error) {
      console.error('세션스토리지 저장 오류:', error);
    }
  };

  const removeValue = () => {
    try {
      sessionStorage.removeItem(key);
      setValueState(null);
    } catch (error) {
      console.error('세션스토리지 삭제 오류:', error);
    }
  };

  return {
    value,
    setValue,
    removeValue,
    isInitialized
  };
}

/**
 * 방문 체크 전용 커스텀 훅
 * 
 * @returns 방문 여부와 관련 함수들
 * 
 * @example
 * ```tsx
 * const { hasVisited, setVisited, isInitialized } = useVisitCheck();
 * 
 * if (!isInitialized) {
 *   return <LoadingSpinner />;
 * }
 * 
 * return hasVisited ? <HomePage /> : <LetsRunPage />;
 * ```
 */
export function useVisitCheck() {
  const { value, setValue, removeValue, isInitialized } = useSessionStorage({
    key: 'hasVisited',
    defaultValue: 'false'
  });

  const hasVisited = value === 'true';
  
  const setVisited = () => setValue('true');
  const resetVisit = () => setValue('false');
  const clearVisit = () => removeValue();

  return {
    hasVisited,
    setVisited,
    resetVisit,
    clearVisit,
    isInitialized
  };
} 