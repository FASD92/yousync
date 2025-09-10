import { useState, useCallback, useRef } from 'react';

/**
 * 더빙 컨테이너의 모든 상태를 정의하는 인터페이스
 * 
 * @interface DubbingState
 * @property {boolean} isSidebarOpen - 사이드바 열림/닫힘 상태 (사이드바 버튼 클릭 시 변경)
 * @property {boolean} showCompleted - 분석 완료 표시 상태 (모든 문장 분석 완료 시 true)
 * @property {boolean} showResults - 결과 섹션 표시 상태 (결과 보기 버튼 클릭 시 true)
 * @property {number} currentScriptIndex - 현재 선택된 스크립트 인덱스 (비디오 시간에 따라 자동 변경)
 * @property {number} currentVideoTime - 현재 비디오 재생 시간 (VideoPlayer에서 실시간 업데이트)
 * @property {boolean} isVideoPlaying - 비디오 재생/일시정지 상태 (재생/일시정지 버튼 클릭 시 변경)
 * @property {Record<string, any>} finalResults - jobId별 분석 결과 저장 (SSE로 실시간 수신)
 * @property {Record<string, any>} latestResultByScript - 스크립트별 최신 분석 결과 (문장별 결과 표시용)
 * @property {boolean} recording - 녹음 진행 상태 (PitchComparison에서 녹음 시작/종료 시 변경)
 * @property {boolean} recordingCompleted - 녹음 완료 상태 (녹음 종료 후 3초간 분석 애니메이션용)
 */
interface DubbingState {
  // UI 상태
  isSidebarOpen: boolean;
  showCompleted: boolean;
  showResults: boolean;
  
  // 비디오/스크립트 상태
  currentScriptIndex: number;
  currentVideoTime: number;
  isVideoPlaying: boolean;
  
  // 분석 결과 상태
  finalResults: Record<string, any>;
  latestResultByScript: Record<string, any>;
  
  // 녹음 상태
  recording: boolean;
  recordingCompleted: boolean;
  
  // 분석 상태
  isAnalyzing: boolean;
}

/**
 * 더빙 컨테이너의 모든 액션을 정의하는 인터페이스
 * 
 * @interface DubbingActions
 * @property {Function} setIsSidebarOpen - 사이드바 열림/닫힘 설정 (사이드바 버튼 클릭 시 호출)
 * @property {Function} setShowCompleted - 분석 완료 표시 설정 (SSE 완료 감지 시 호출)
 * @property {Function} setShowResults - 결과 섹션 표시 설정 (결과 보기 버튼 클릭 시 호출)
 * @property {Function} setCurrentScriptIndex - 현재 스크립트 인덱스 설정 (비디오 시간 변경 시 자동 호출)
 * @property {Function} setCurrentVideoTime - 현재 비디오 시간 설정 (VideoPlayer에서 실시간 호출)
 * @property {Function} setIsVideoPlaying - 비디오 재생 상태 설정 (재생/일시정지 버튼 클릭 시 호출)
 * @property {Function} setFinalResults - 전체 분석 결과 설정 (SSE 결과 수신 시 호출)
 * @property {Function} setLatestResultByScript - 스크립트별 결과 설정 (SSE 결과 수신 시 호출)
 * @property {Function} addFinalResult - jobId별 결과 추가 (SSE completed 이벤트 시 호출)
 * @property {Function} addLatestResultByScript - 스크립트별 결과 추가 (SSE completed 이벤트 시 호출)
 * @property {Function} setRecording - 녹음 상태 설정 (PitchComparison에서 녹음 시작/종료 시 호출)
 * @property {Function} setRecordingCompleted - 녹음 완료 상태 설정 (녹음 종료 시 호출)
 * @property {Function} handleRecordingComplete - 녹음 완료 핸들러 (녹음 종료 후 3초간 분석 애니메이션 시작)
 * @property {Function} nextScript - 다음 스크립트로 이동 (스크립트 네비게이션 버튼 클릭 시 호출)
 * @property {Function} prevScript - 이전 스크립트로 이동 (스크립트 네비게이션 버튼 클릭 시 호출)
 * @property {Function} resetState - 모든 상태 초기화 (새로운 분석 시작 시 호출)
 * @property {Function} handlePlay - 비디오 재생 핸들러 (VideoPlayer 재생 버튼 클릭 시 호출)
 * @property {Function} handlePause - 비디오 일시정지 핸들러 (VideoPlayer 일시정지 버튼 클릭 시 호출)
 * @property {Function} handleScriptSelect - 스크립트 선택 핸들러 (사이드바에서 스크립트 클릭 시 호출)
 */
interface DubbingActions {
  // UI 액션
  setIsSidebarOpen: (open: boolean) => void;
  setShowCompleted: (completed: boolean) => void;
  setShowResults: (show: boolean) => void;
  
  // 비디오/스크립트 액션
  setCurrentScriptIndex: (index: number) => void;
  setCurrentVideoTime: (time: number) => void;
  setIsVideoPlaying: (playing: boolean) => void;
  
  // 분석 결과 액션
  setFinalResults: (results: Record<string, any>) => void;
  setLatestResultByScript: (results: Record<string, any>) => void;
  addFinalResult: (jobId: string, result: any) => void;
  addLatestResultByScript: (scriptKey: string, result: any) => void;
  
  // 녹음 액션
  setRecording: (recording: boolean) => void;
  setRecordingCompleted: (completed: boolean) => void;
  handleRecordingComplete: () => void;
  
  // 분석 액션
  setIsAnalyzing: (analyzing: boolean) => void;
  
  // 편의 메서드
  nextScript: () => void;
  prevScript: () => void;
  resetState: () => void;
  
  // 비디오 제어
  handlePlay: () => void;
  handlePause: () => void;
  handleScriptSelect: (index: number) => void;
}

/**
 * 더빙 컨테이너의 모든 상태와 액션을 관리하는 커스텀 훅
 * 
 * @param {number} totalScripts - 전체 스크립트 개수 (front_data.captions.length)
 * @param {Object} options - 외부 콜백 함수들
 * @param {Function} options.onScriptChange - 스크립트 변경 시 호출될 콜백 (스크립트 인덱스 변경 시 추가 로직 실행)
 * @param {Function} options.onPlay - 비디오 재생 시 호출될 콜백 (재생 시 추가 로직 실행)
 * @param {Function} options.onPause - 비디오 일시정지 시 호출될 콜백 (일시정지 시 추가 로직 실행)
 * @param {Function} options.onRecordingChange - 녹음 상태 변경 시 호출될 콜백 (녹음 시작/종료 시 추가 로직 실행)
 * @param {boolean} initialSidebarOpen - 초기 사이드바 열림 상태 (기본값: false)
 * 
 * @returns {DubbingState & DubbingActions} 모든 상태와 액션을 포함한 객체
 * 
 * @example
 * ```tsx
 * const dubbingState = useDubbingState(front_data.captions.length, {
 *   onScriptChange: (index) => {
 *     // 스크립트 변경 시 추가 로직
 *     console.log('스크립트 변경:', index);
 *   },
 *   onPlay: () => {
 *     // 재생 시 추가 로직
 *     console.log('비디오 재생');
 *   }
 * }, true); // 초기에 사이드바 열림
 * 
 * const { currentScriptIndex, handlePlay, setRecording } = dubbingState;
 * ```
 * 
 * @description
 * 이 훅은 DubbingContainer에서 사용되는 모든 상태를 중앙에서 관리합니다.
 * - UI 상태: 사이드바, 완료 표시, 결과 표시
 * - 비디오/스크립트 상태: 현재 스크립트, 비디오 시간, 재생 상태
 * - 분석 결과 상태: jobId별 결과, 스크립트별 결과
 * - 녹음 상태: 녹음 진행 여부
 * 
 * 주요 사용 시나리오:
 * 1. VideoPlayer에서 시간 업데이트 → setCurrentVideoTime 호출
 * 2. SSE에서 분석 결과 수신 → addFinalResult, addLatestResultByScript 호출
 * 3. 사용자가 재생/일시정지 버튼 클릭 → handlePlay/handlePause 호출
 * 4. 사이드바에서 스크립트 선택 → handleScriptSelect 호출
 * 5. 모든 분석 완료 감지 → setShowCompleted(true) 호출
 */
export function useDubbingState(
  totalScripts: number = 0,
  options?: {
    onScriptChange?: (index: number) => void;
    onPlay?: () => void;
    onPause?: () => void;
    onRecordingChange?: (recording: boolean) => void;
  },
  initialSidebarOpen: boolean = false
): DubbingState & DubbingActions {
  // UI 상태
  const [isSidebarOpen, setIsSidebarOpen] = useState(initialSidebarOpen);
  const [showCompleted, setShowCompleted] = useState(false);
  const [showResults, setShowResults] = useState(false);
  
  // 비디오/스크립트 상태
  const [currentScriptIndex, setCurrentScriptIndex] = useState(0);
  const [currentVideoTime, setCurrentVideoTime] = useState(0);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  
  // 분석 결과 상태
  const [finalResults, setFinalResults] = useState<Record<string, any>>({});
  const [latestResultByScript, setLatestResultByScript] = useState<Record<string, any>>({});
  
  // 녹음 상태
  const [recording, setRecording] = useState(false);
  const [recordingCompleted, setRecordingCompleted] = useState(false);
  
  // 분석 상태
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // 외부 콜백들
  const { onScriptChange, onPlay, onPause, onRecordingChange } = options || {};
  
  /**
   * 다음 스크립트로 이동하는 편의 함수
   * 
   * @description
   * - 현재 스크립트 인덱스를 1 증가시킴
   * - 전체 스크립트 개수를 초과하지 않도록 제한
   * - 스크립트 변경 시 외부 콜백 호출
   * 
   * 사용 시점: ScriptDisplay의 다음 버튼 클릭 시
   */
  const nextScript = useCallback(() => {
    const nextIndex = Math.min(currentScriptIndex + 1, totalScripts - 1);
    setCurrentScriptIndex(nextIndex);
    onScriptChange?.(nextIndex);
  }, [currentScriptIndex, totalScripts, onScriptChange]);
  
  /**
   * 이전 스크립트로 이동하는 편의 함수
   * 
   * @description
   * - 현재 스크립트 인덱스를 1 감소시킴
   * - 0 미만으로 내려가지 않도록 제한
   * - 스크립트 변경 시 외부 콜백 호출
   * 
   * 사용 시점: ScriptDisplay의 이전 버튼 클릭 시
   */
  const prevScript = useCallback(() => {
    const prevIndex = Math.max(currentScriptIndex - 1, 0);
    setCurrentScriptIndex(prevIndex);
    onScriptChange?.(prevIndex);
  }, [currentScriptIndex, onScriptChange]);
  
  /**
   * jobId별 분석 결과를 추가하는 함수
   * 
   * @param {string} jobId - 분석 작업 ID
   * @param {any} result - 분석 결과 데이터
   * 
   * @description
   * - SSE에서 'completed' 이벤트 수신 시 호출
   * - 진행상황 추적을 위해 jobId별로 결과 저장
   * - 기존 결과를 유지하면서 새로운 결과 추가
   * 
   * 사용 시점: SSE onmessage 이벤트에서 data.status === 'completed' 시
   */
  const addFinalResult = useCallback((jobId: string, result: any) => {
    setFinalResults((prev: Record<string, any>) => ({
      ...prev,
      [jobId]: result
    }));
  }, []);
  
  /**
   * 스크립트별 최신 분석 결과를 추가하는 함수
   * 
   * @param {string} scriptKey - 정규화된 스크립트 텍스트 (키로 사용)
   * @param {any} result - 분석 결과 데이터
   * 
   * @description
   * - SSE에서 'completed' 이벤트 수신 시 호출
   * - 문장별 결과 표시를 위해 스크립트별로 결과 저장
   * - 동일한 스크립트에 대해 최신 결과만 유지
   * 
   * 사용 시점: SSE onmessage 이벤트에서 data.status === 'completed' 시
   */
  const addLatestResultByScript = useCallback((scriptKey: string, result: any) => {
    setLatestResultByScript((prev: Record<string, any>) => ({
      ...prev,
      [scriptKey]: result
    }));
  }, []);
  
  /**
   * 비디오 재생 핸들러
   * 
   * @description
   * - 비디오 재생 상태를 true로 설정
   * - 외부 재생 콜백 호출 (추가 로직 실행)
   * 
   * 사용 시점: VideoPlayer 재생 버튼 클릭 시
   */
  const handlePlay = useCallback(() => {
    setIsVideoPlaying(true);
    onPlay?.();
  }, [onPlay]);
  
  /**
   * 비디오 일시정지 핸들러
   * 
   * @description
   * - 비디오 재생 상태를 false로 설정
   * - 외부 일시정지 콜백 호출 (추가 로직 실행)
   * 
   * 사용 시점: VideoPlayer 일시정지 버튼 클릭 시
   */
  const handlePause = useCallback(() => {
    setIsVideoPlaying(false);
    onPause?.();
  }, [onPause]);
  
  /**
   * 스크립트 선택 핸들러
   * 
   * @param {number} index - 선택할 스크립트 인덱스
   * 
   * @description
   * - 현재 스크립트 인덱스를 지정된 인덱스로 변경
   * - 외부 스크립트 변경 콜백 호출 (추가 로직 실행)
   * 
   * 사용 시점: Sidebar에서 스크립트 클릭 시
   */
  const handleScriptSelect = useCallback((index: number) => {
    setCurrentScriptIndex(index);
    onScriptChange?.(index);
  }, [onScriptChange]);
  
  /**
   * 모든 상태를 초기값으로 리셋하는 함수
   * 
   * @description
   * - 모든 상태를 초기값으로 되돌림
   * - 새로운 분석 시작 시 사용
   * - 스크립트 인덱스: 0, 비디오 시간: 0, 재생 상태: false
   * - 완료/결과 표시: false, 분석 결과: 빈 객체, 녹음: false
   * 
   * 사용 시점: 새로운 분석 시작 시
   */
  const resetState = useCallback(() => {
    setCurrentScriptIndex(0);
    setCurrentVideoTime(0);
    setIsVideoPlaying(false);
    setShowCompleted(false);
    setShowResults(false);
    setFinalResults({});
    setLatestResultByScript({});
    setRecording(false);
    setRecordingCompleted(false);
  }, []);
  
  /**
   * 녹음 완료 핸들러
   * 
   * @description
   * - 녹음 완료 상태를 true로 설정
   * - 분석 결과가 들어올 때까지 유지
   * 
   * 사용 시점: PitchComparison에서 녹음 종료 시
   */
  const handleRecordingComplete = useCallback(() => {
    console.log('[DEBUG] handleRecordingComplete 호출됨 - recordingCompleted를 true로 설정');
    setRecordingCompleted(true);
    // 자동으로 false로 변경하지 않음 - 분석 결과가 들어올 때까지 유지
  }, []);
  
  return {
    // 상태
    isSidebarOpen,
    showCompleted,
    showResults,
    currentScriptIndex,
    currentVideoTime,
    isVideoPlaying,
    finalResults,
    latestResultByScript,
    recording,
    recordingCompleted,
    isAnalyzing,
    
    // 액션
    setIsSidebarOpen,
    setShowCompleted,
    setShowResults,
    setCurrentScriptIndex,
    setCurrentVideoTime,
    setIsVideoPlaying,
    setFinalResults,
    setLatestResultByScript,
    addFinalResult,
    addLatestResultByScript,
    setRecording,
    setRecordingCompleted,
    setIsAnalyzing,
    handleRecordingComplete,
    nextScript,
    prevScript,
    resetState,
    handlePlay,
    handlePause,
    handleScriptSelect,
  };
} 