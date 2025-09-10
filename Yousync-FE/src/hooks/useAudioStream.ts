import { useEffect } from "react";
import { useAudioStore } from "@/store/useAudioStore";
import toast from 'react-hot-toast';

export function useAudioStream() {
  const initMic = async () => {
    try {
      console.log('마이크 권한 요청 중...');
      // 기존 스트림이 있다면 정리
      const currentState = useAudioStore.getState();
      if (currentState.stream) {
        currentState.stream.getTracks().forEach(track => track.stop());
      }
      if (currentState.audioCtx) {
        currentState.audioCtx.close();
      }
      useAudioStore.getState().reset();
      // 권한 상태 확인
      const permission = await navigator.permissions.query({name: 'microphone'});
      console.log('현재 마이크 권한 상태:', permission.state);
      if (permission.state === 'denied') {
        toast('마이크 권한이 완전히 차단되어 있습니다.\n\n브라우저 설정에서 마이크 권한을 허용해주세요:\n1. 주소창 왼쪽 마이크 아이콘 클릭\n2. "허용" 선택\n3. 페이지 새로고침');
        return;
      }
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: false,
          noiseSuppression: true,
          autoGainControl: false,
          sampleRate: 48000,
          channelCount: 1
        } 
      });
      const audioCtx = new AudioContext();
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);
      useAudioStore.getState().setAudioCtx({ stream, audioCtx, analyser, source });
    } catch (error) {
      console.error('마이크 권한이 거부되었습니다:', error);
      toast('마이크 권한이 거부되었습니다.\n\n브라우저 주소창 왼쪽의 마이크 아이콘을 클릭하여 권한을 허용해주세요.');
    }
  };

  // 마이크 정리 함수
  const cleanupMic = () => {
    console.log('마이크 정리 중...');
    const { audioCtx, stream } = useAudioStore.getState();
    
    // 오디오 컨텍스트 정리
    if (audioCtx) {
      try {
        audioCtx.close();
        console.log('오디오 컨텍스트 정리 완료');
      } catch (error) {
        console.error('오디오 컨텍스트 정리 중 오류:', error);
      }
    }
    
    // 미디어 스트림 정리
    if (stream) {
      try {
        stream.getTracks().forEach(track => {
          track.stop();
          console.log('마이크 트랙 정지:', track.kind);
        });
        console.log('미디어 스트림 정리 완료');
      } catch (error) {
        console.error('미디어 스트림 정리 중 오류:', error);
      }
    }
    
    // 전역 상태 초기화
    useAudioStore.getState().reset();
    console.log('마이크 정리 완료');
  };

  useEffect(() => {
    initMic();
    
    // 컴포넌트 언마운트 시 마이크 정리
    return () => {
      cleanupMic();
    };
  }, []);

  // 페이지를 벗어날 때 마이크 정리 (beforeunload 이벤트)
  useEffect(() => {
    const handleBeforeUnload = () => {
      console.log('페이지를 벗어남 - 마이크 정리');
      cleanupMic();
    };

    const handleVisibilityChange = () => {
      if (document.hidden) {
        console.log('페이지가 숨겨짐 - 마이크 정리');
        cleanupMic();
      }
    };

    // 페이지를 벗어날 때
    window.addEventListener('beforeunload', handleBeforeUnload);
    
    // 탭이 숨겨질 때 (다른 탭으로 이동)
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // 브라우저 뒤로가기/앞으로가기
    window.addEventListener('popstate', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('popstate', handleBeforeUnload);
    };
  }, []);

  return { initMic, cleanupMic };
}