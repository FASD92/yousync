// 스크립트별 배경음 관리 유틸리티

export interface BackgroundAudioConfig {
  scriptId: number;
  audioUrl: string;
  startTime: number;
  endTime: number;
  volume?: number;
}

export interface ScriptBackgroundAudio {
  id: number;
  background_audio_url: string;
  start_time: number;
  end_time: number;
  script: string;
}

export interface S3BackgroundAudio {
  scriptId: number;
  s3Url: string;
  script: string;
  index: number;
}

/**
 * 스크립트별 배경음 URL을 추출하는 함수
 * 다양한 구조에서 사용자 오디오 URL 추출 함수와 유사한 패턴으로 구현
 */
export function extractScriptBackgroundAudioUrl(script: any): string | null {
  if (!script) return null;
  
  return (
    script.background_audio_url ||
    script.bgvoice_url ||
    script.background_audio ||
    script.audio_url ||
    (script.background_audio && script.background_audio.url) ||
    (script.background_audio && script.background_audio.s3_url) ||
    null
  );
}

/**
 * S3 배경음 URL 배열을 스크립트별 배경음 정보로 변환
 */
export function convertS3AudiosToBackgroundAudio(scriptAudios: string[], scripts?: any[]): S3BackgroundAudio[] {
  return scriptAudios.map((s3Url, index) => ({
    scriptId: scripts?.[index]?.id || index,
    s3Url,
    script: scripts?.[index]?.script || `스크립트 ${index + 1}`,
    index
  }));
}

/**
 * 스크립트 배열에서 배경음 정보를 추출하는 함수
 */
export function extractScriptsBackgroundAudio(scripts: any[]): ScriptBackgroundAudio[] {
  if (!Array.isArray(scripts)) return [];
  
  return scripts.map((script, index) => ({
    id: script.id || index,
    background_audio_url: extractScriptBackgroundAudioUrl(script) || '',
    start_time: script.start_time || 0,
    end_time: script.end_time || 0,
    script: script.script || ''
  })).filter(item => item.background_audio_url);
}

/**
 * 배경음 오디오 요소를 생성하고 관리하는 클래스
 */
export class BackgroundAudioManager {
  private audioElements: Map<number, HTMLAudioElement> = new Map();
  private configs: Map<number, BackgroundAudioConfig> = new Map();
  private s3AudioElements: Map<number, HTMLAudioElement> = new Map();

  /**
   * 스크립트별 배경음 설정
   */
  setScriptBackgroundAudio(config: BackgroundAudioConfig): void {
    const { scriptId, audioUrl, startTime, endTime, volume = 0.5 } = config;
    
    // 기존 오디오 요소 정리
    this.removeScriptBackgroundAudio(scriptId);
    
    // 새 오디오 요소 생성
    const audio = new Audio(audioUrl);
    audio.volume = volume;
    audio.preload = 'auto';
    
    this.audioElements.set(scriptId, audio);
    this.configs.set(scriptId, config);
    
    console.log(`[BackgroundAudio] 스크립트 ${scriptId} 배경음 설정됨:`, {
      url: audioUrl,
      startTime,
      endTime,
      volume
    });
  }

  /**
   * S3 배경음 설정
   */
  setS3BackgroundAudio(scriptId: number, s3Url: string, volume: number = 0.5): void {
    // 기존 S3 오디오 요소 정리
    this.removeS3BackgroundAudio(scriptId);
    
    // 새 오디오 요소 생성
    const audio = new Audio(s3Url);
    audio.volume = volume;
    audio.preload = 'auto';
    
    this.s3AudioElements.set(scriptId, audio);
    
    console.log(`[BackgroundAudio] 스크립트 ${scriptId} S3 배경음 설정됨:`, {
      url: s3Url,
      volume
    });
  }

  /**
   * 특정 스크립트의 배경음 재생
   */
  playScriptBackgroundAudio(scriptId: number, currentTime: number = 0): Promise<void> {
    const audio = this.audioElements.get(scriptId);
    const config = this.configs.get(scriptId);
    
    if (!audio || !config) {
      console.warn(`[BackgroundAudio] 스크립트 ${scriptId}의 배경음이 설정되지 않았습니다.`);
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      // 오디오 로드 완료 후 재생
      if (audio.readyState >= 2) {
        this.startPlayback(audio, config, currentTime);
        resolve();
      } else {
        audio.addEventListener('canplaythrough', () => {
          this.startPlayback(audio, config, currentTime);
          resolve();
        }, { once: true });
        
        audio.addEventListener('error', (error) => {
          console.error(`[BackgroundAudio] 스크립트 ${scriptId} 배경음 로드 실패:`, error);
          reject(error);
        });
      }
    });
  }

  /**
   * S3 배경음 재생
   */
  playS3BackgroundAudio(scriptId: number): Promise<void> {
    const audio = this.s3AudioElements.get(scriptId);
    
    if (!audio) {
      console.warn(`[BackgroundAudio] 스크립트 ${scriptId}의 S3 배경음이 설정되지 않았습니다.`);
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      if (audio.readyState >= 2) {
        audio.play().then(() => {
          console.log(`[BackgroundAudio] 스크립트 ${scriptId} S3 배경음 재생 시작`);
          resolve();
        }).catch((error) => {
          console.error(`[BackgroundAudio] 스크립트 ${scriptId} S3 배경음 재생 실패:`, error);
          reject(error);
        });
      } else {
        audio.addEventListener('canplaythrough', () => {
          audio.play().then(() => {
            console.log(`[BackgroundAudio] 스크립트 ${scriptId} S3 배경음 재생 시작`);
            resolve();
          }).catch((error) => {
            console.error(`[BackgroundAudio] 스크립트 ${scriptId} S3 배경음 재생 실패:`, error);
            reject(error);
          });
        }, { once: true });
        
        audio.addEventListener('error', (error) => {
          console.error(`[BackgroundAudio] 스크립트 ${scriptId} S3 배경음 로드 실패:`, error);
          reject(error);
        });
      }
    });
  }

  /**
   * 배경음 재생 시작
   */
  private startPlayback(audio: HTMLAudioElement, config: BackgroundAudioConfig, currentTime: number): void {
    const { startTime, endTime } = config;
    
    // 현재 시간에 맞춰 오디오 시작 시간 조정
    const audioStartTime = Math.max(0, currentTime - startTime);
    audio.currentTime = audioStartTime;
    
    // 재생 시작
    audio.play().then(() => {
      console.log(`[BackgroundAudio] 스크립트 ${config.scriptId} 배경음 재생 시작`);
      
      // 종료 시간에 맞춰 정지
      const duration = endTime - startTime;
      setTimeout(() => {
        this.stopScriptBackgroundAudio(config.scriptId);
      }, duration * 1000);
      
    }).catch((error) => {
      console.error(`[BackgroundAudio] 스크립트 ${config.scriptId} 배경음 재생 실패:`, error);
    });
  }

  /**
   * 특정 스크립트의 배경음 정지
   */
  stopScriptBackgroundAudio(scriptId: number): void {
    const audio = this.audioElements.get(scriptId);
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
      console.log(`[BackgroundAudio] 스크립트 ${scriptId} 배경음 정지`);
    }
  }

  /**
   * S3 배경음 정지
   */
  stopS3BackgroundAudio(scriptId: number): void {
    const audio = this.s3AudioElements.get(scriptId);
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
      console.log(`[BackgroundAudio] 스크립트 ${scriptId} S3 배경음 정지`);
    }
  }

  /**
   * 특정 스크립트의 배경음 제거
   */
  removeScriptBackgroundAudio(scriptId: number): void {
    const audio = this.audioElements.get(scriptId);
    if (audio) {
      audio.pause();
      audio.src = '';
      this.audioElements.delete(scriptId);
      this.configs.delete(scriptId);
      console.log(`[BackgroundAudio] 스크립트 ${scriptId} 배경음 제거`);
    }
  }

  /**
   * S3 배경음 제거
   */
  removeS3BackgroundAudio(scriptId: number): void {
    const audio = this.s3AudioElements.get(scriptId);
    if (audio) {
      audio.pause();
      audio.src = '';
      this.s3AudioElements.delete(scriptId);
      console.log(`[BackgroundAudio] 스크립트 ${scriptId} S3 배경음 제거`);
    }
  }

  /**
   * 모든 배경음 정지
   */
  stopAllBackgroundAudio(): void {
    this.audioElements.forEach((audio, scriptId) => {
      audio.pause();
      audio.currentTime = 0;
    });
    this.s3AudioElements.forEach((audio, scriptId) => {
      audio.pause();
      audio.currentTime = 0;
    });
    console.log('[BackgroundAudio] 모든 배경음 정지');
  }

  /**
   * 모든 배경음 제거
   */
  removeAllBackgroundAudio(): void {
    this.audioElements.forEach((audio) => {
      audio.pause();
      audio.src = '';
    });
    this.s3AudioElements.forEach((audio) => {
      audio.pause();
      audio.src = '';
    });
    this.audioElements.clear();
    this.s3AudioElements.clear();
    this.configs.clear();
    console.log('[BackgroundAudio] 모든 배경음 제거');
  }

  /**
   * 현재 재생 중인 스크립트 ID들 반환
   */
  getPlayingScriptIds(): number[] {
    const playingScripts = Array.from(this.audioElements.entries())
      .filter(([_, audio]) => !audio.paused)
      .map(([scriptId, _]) => scriptId);
    
    const playingS3Scripts = Array.from(this.s3AudioElements.entries())
      .filter(([_, audio]) => !audio.paused)
      .map(([scriptId, _]) => scriptId);
    
    return [...playingScripts, ...playingS3Scripts];
  }

  /**
   * 배경음 볼륨 설정
   */
  setVolume(scriptId: number, volume: number): void {
    const audio = this.audioElements.get(scriptId);
    if (audio) {
      audio.volume = Math.max(0, Math.min(1, volume));
      console.log(`[BackgroundAudio] 스크립트 ${scriptId} 볼륨 설정: ${volume}`);
    }
  }

  /**
   * S3 배경음 볼륨 설정
   */
  setS3Volume(scriptId: number, volume: number): void {
    const audio = this.s3AudioElements.get(scriptId);
    if (audio) {
      audio.volume = Math.max(0, Math.min(1, volume));
      console.log(`[BackgroundAudio] 스크립트 ${scriptId} S3 볼륨 설정: ${volume}`);
    }
  }

  /**
   * 모든 배경음 볼륨 설정
   */
  setAllVolumes(volume: number): void {
    this.audioElements.forEach((audio, scriptId) => {
      audio.volume = Math.max(0, Math.min(1, volume));
    });
    this.s3AudioElements.forEach((audio, scriptId) => {
      audio.volume = Math.max(0, Math.min(1, volume));
    });
    console.log(`[BackgroundAudio] 모든 배경음 볼륨 설정: ${volume}`);
  }

  /**
   * S3 배경음이 설정되어 있는지 확인
   */
  hasS3BackgroundAudio(scriptId: number): boolean {
    return this.s3AudioElements.has(scriptId);
  }

  /**
   * 일반 배경음이 설정되어 있는지 확인
   */
  hasBackgroundAudio(scriptId: number): boolean {
    return this.audioElements.has(scriptId);
  }
}

/**
 * 전역 배경음 매니저 인스턴스
 */
export const backgroundAudioManager = new BackgroundAudioManager(); 