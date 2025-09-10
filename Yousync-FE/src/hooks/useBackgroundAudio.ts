import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  backgroundAudioManager, 
  BackgroundAudioConfig, 
  ScriptBackgroundAudio,
  extractScriptsBackgroundAudio 
} from '@/utils/backgroundAudio';

interface UseBackgroundAudioProps {
  scripts?: any[];
  captions?: any[];
  scriptAudios?: string[]; // S3 배경음 URL 배열
  fullAudio?: string;      // 전체 S3 배경음 URL
  onBackgroundAudioStart?: (scriptId: number) => void;
  onBackgroundAudioEnd?: (scriptId: number) => void;
  onBackgroundAudioError?: (scriptId: number, error: any) => void;
}

interface UseBackgroundAudioReturn {
  // 상태
  isBackgroundAudioPlaying: boolean;
  currentPlayingScriptId: number | null;
  backgroundAudioVolume: number;
  
  // 메서드
  setupScriptBackgroundAudio: (scriptId: number, config: Omit<BackgroundAudioConfig, 'scriptId'>) => void;
  playScriptBackgroundAudio: (scriptId: number, currentTime?: number) => Promise<void>;
  stopScriptBackgroundAudio: (scriptId: number) => void;
  stopAllBackgroundAudio: () => void;
  setBackgroundAudioVolume: (volume: number) => void;
  setScriptBackgroundAudioVolume: (scriptId: number, volume: number) => void;
  
  // 유틸리티
  getBackgroundAudioConfigs: () => ScriptBackgroundAudio[];
  isScriptBackgroundAudioReady: (scriptId: number) => boolean;
}

export function useBackgroundAudio({
  scripts = [],
  captions = [],
  scriptAudios = [],
  fullAudio,
  onBackgroundAudioStart,
  onBackgroundAudioEnd,
  onBackgroundAudioError
}: UseBackgroundAudioProps = {}): UseBackgroundAudioReturn {
  
  const [isBackgroundAudioPlaying, setIsBackgroundAudioPlaying] = useState(false);
  const [currentPlayingScriptId, setCurrentPlayingScriptId] = useState<number | null>(null);
  const [backgroundAudioVolume, setBackgroundAudioVolumeState] = useState(0.5);
  
  const audioEventListeners = useRef<Map<number, { start: () => void; end: () => void; error: (e: Event) => void }>>(new Map());

  // 스크립트 데이터에서 배경음 정보 추출
  const backgroundAudioConfigs = useCallback(() => {
    const scriptAudios = extractScriptsBackgroundAudio(scripts);
    const captionAudios = extractScriptsBackgroundAudio(captions);
    
    // 중복 제거하여 합치기
    const allAudios = [...scriptAudios, ...captionAudios];
    const uniqueAudios = allAudios.filter((audio, index, self) => 
      index === self.findIndex(a => a.id === audio.id)
    );
    
    return uniqueAudios;
  }, [scripts, captions]);

  // S3 배경음 설정
  const setupS3BackgroundAudio = useCallback(() => {
    // 스크립트별 S3 배경음 설정
    scriptAudios.forEach((s3Url, index) => {
      const scriptId = scripts?.[index]?.id || index;
      backgroundAudioManager.setS3BackgroundAudio(scriptId, s3Url, backgroundAudioVolume);
    });

    // 전체 S3 배경음 설정 (ID를 -1로 설정하여 구분)
    if (fullAudio) {
      backgroundAudioManager.setS3BackgroundAudio(-1, fullAudio, backgroundAudioVolume);
    }
  }, [scriptAudios, fullAudio, scripts, backgroundAudioVolume]);

  // 스크립트별 배경음 설정
  const setupScriptBackgroundAudio = useCallback((scriptId: number, config: Omit<BackgroundAudioConfig, 'scriptId'>) => {
    const fullConfig: BackgroundAudioConfig = {
      scriptId,
      ...config,
      volume: config.volume ?? backgroundAudioVolume
    };
    
    backgroundAudioManager.setScriptBackgroundAudio(fullConfig);
    
    // 이벤트 리스너 설정
    const audio = backgroundAudioManager['audioElements'].get(scriptId);
    if (audio) {
      // 기존 리스너 제거
      const existingListeners = audioEventListeners.current.get(scriptId);
      if (existingListeners) {
        audio.removeEventListener('play', existingListeners.start);
        audio.removeEventListener('ended', existingListeners.end);
        audio.removeEventListener('error', existingListeners.error);
      }
      
      // 새 리스너 설정
      const listeners = {
        start: () => {
          setIsBackgroundAudioPlaying(true);
          setCurrentPlayingScriptId(scriptId);
          onBackgroundAudioStart?.(scriptId);
        },
        end: () => {
          if (currentPlayingScriptId === scriptId) {
            setIsBackgroundAudioPlaying(false);
            setCurrentPlayingScriptId(null);
          }
          onBackgroundAudioEnd?.(scriptId);
        },
        error: (e: Event) => {
          console.error(`[BackgroundAudio] 스크립트 ${scriptId} 배경음 에러:`, e);
          onBackgroundAudioError?.(scriptId, e);
        }
      };
      
      audio.addEventListener('play', listeners.start);
      audio.addEventListener('ended', listeners.end);
      audio.addEventListener('error', listeners.error);
      
      audioEventListeners.current.set(scriptId, listeners);
    }
  }, [backgroundAudioVolume, currentPlayingScriptId, onBackgroundAudioStart, onBackgroundAudioEnd, onBackgroundAudioError]);

  // 스크립트별 배경음 재생
  const playScriptBackgroundAudio = useCallback(async (scriptId: number, currentTime: number = 0) => {
    try {
      // 일반 배경음이 있으면 재생
      if (backgroundAudioManager.hasBackgroundAudio(scriptId)) {
        await backgroundAudioManager.playScriptBackgroundAudio(scriptId, currentTime);
      }
      // S3 배경음이 있으면 재생
      else if (backgroundAudioManager.hasS3BackgroundAudio(scriptId)) {
        await backgroundAudioManager.playS3BackgroundAudio(scriptId);
      }
    } catch (error) {
      console.error(`[BackgroundAudio] 스크립트 ${scriptId} 배경음 재생 실패:`, error);
      onBackgroundAudioError?.(scriptId, error);
    }
  }, [onBackgroundAudioError]);

  // 스크립트별 배경음 정지
  const stopScriptBackgroundAudio = useCallback((scriptId: number) => {
    backgroundAudioManager.stopScriptBackgroundAudio(scriptId);
    backgroundAudioManager.stopS3BackgroundAudio(scriptId);
    
    if (currentPlayingScriptId === scriptId) {
      setIsBackgroundAudioPlaying(false);
      setCurrentPlayingScriptId(null);
    }
  }, [currentPlayingScriptId]);

  // 모든 배경음 정지
  const stopAllBackgroundAudio = useCallback(() => {
    backgroundAudioManager.stopAllBackgroundAudio();
    setIsBackgroundAudioPlaying(false);
    setCurrentPlayingScriptId(null);
  }, []);

  // 전체 배경음 볼륨 설정
  const setBackgroundAudioVolume = useCallback((volume: number) => {
    const clampedVolume = Math.max(0, Math.min(1, volume));
    setBackgroundAudioVolumeState(clampedVolume);
    backgroundAudioManager.setAllVolumes(clampedVolume);
  }, []);

  // 특정 스크립트 배경음 볼륨 설정
  const setScriptBackgroundAudioVolume = useCallback((scriptId: number, volume: number) => {
    backgroundAudioManager.setVolume(scriptId, volume);
  }, []);

  // 배경음 설정 정보 반환
  const getBackgroundAudioConfigs = useCallback(() => {
    return backgroundAudioConfigs();
  }, [backgroundAudioConfigs]);

  // 스크립트 배경음 준비 상태 확인
  const isScriptBackgroundAudioReady = useCallback((scriptId: number) => {
    return backgroundAudioManager.hasBackgroundAudio(scriptId) || backgroundAudioManager.hasS3BackgroundAudio(scriptId);
  }, []);

  // 컴포넌트 언마운트 시 정리
  useEffect(() => {
    return () => {
      // 모든 이벤트 리스너 제거
      audioEventListeners.current.forEach((listeners, scriptId) => {
        const audio = backgroundAudioManager['audioElements'].get(scriptId);
        if (audio) {
          audio.removeEventListener('play', listeners.start);
          audio.removeEventListener('ended', listeners.end);
          audio.removeEventListener('error', listeners.error);
        }
      });
      audioEventListeners.current.clear();
      
      // 모든 배경음 정리
      backgroundAudioManager.removeAllBackgroundAudio();
    };
  }, []);

  // 스크립트 데이터 변경 시 자동으로 배경음 설정
  useEffect(() => {
    const configs = backgroundAudioConfigs();
    
    configs.forEach((config) => {
      if (config.background_audio_url) {
        setupScriptBackgroundAudio(config.id, {
          audioUrl: config.background_audio_url,
          startTime: config.start_time,
          endTime: config.end_time,
          volume: backgroundAudioVolume
        });
      }
    });

    // S3 배경음 설정
    setupS3BackgroundAudio();
  }, [scripts, captions, scriptAudios, fullAudio, backgroundAudioVolume, setupScriptBackgroundAudio, backgroundAudioConfigs, setupS3BackgroundAudio]);

  return {
    // 상태
    isBackgroundAudioPlaying,
    currentPlayingScriptId,
    backgroundAudioVolume,
    
    // 메서드
    setupScriptBackgroundAudio,
    playScriptBackgroundAudio,
    stopScriptBackgroundAudio,
    stopAllBackgroundAudio,
    setBackgroundAudioVolume,
    setScriptBackgroundAudioVolume,
    
    // 유틸리티
    getBackgroundAudioConfigs,
    isScriptBackgroundAudioReady
  };
} 