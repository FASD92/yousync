// 이벤트 타입 정의
export interface AnalyticsEvent {
  name: string;
  properties?: Record<string, unknown>;
  timestamp?: number;
}

// 사용자 속성 타입
export interface UserProperties {
  userId?: string;
  email?: string;
  name?: string;
  level?: number;
  totalPlays?: number;
}

// 애널리틱스 설정
interface AnalyticsConfig {
  enabled: boolean;
  endpoint?: string;
  batchSize: number;
  flushInterval: number;
}

// 기본 설정
const defaultConfig: AnalyticsConfig = {
  enabled: true,
  batchSize: 10,
  flushInterval: 5000, // 5초
};

class Analytics {
  private config: AnalyticsConfig;
  private events: AnalyticsEvent[] = [];
  private userProperties: UserProperties = {};
  private flushTimer: NodeJS.Timeout | null = null;

  constructor(config: Partial<AnalyticsConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
    this.startFlushTimer();
  }

  // 이벤트 추적
  track(eventName: string, properties?: Record<string, unknown>): void {
    if (!this.config.enabled) return;

    const event: AnalyticsEvent = {
      name: eventName,
      properties: {
        ...properties,
        ...this.userProperties,
      },
      timestamp: Date.now(),
    };

    this.events.push(event);

    // 배치 크기에 도달하면 즉시 전송
    if (this.events.length >= this.config.batchSize) {
      this.flush();
    }
  }

  // 사용자 속성 설정
  identify(properties: UserProperties): void {
    this.userProperties = { ...this.userProperties, ...properties };
  }

  // 페이지 뷰 추적
  pageView(pageName: string, properties?: Record<string, unknown>): void {
    this.track('page_view', {
      page_name: pageName,
      ...properties,
    });
  }

  // 사용자 액션 추적
  action(actionName: string, properties?: Record<string, unknown>): void {
    this.track('user_action', {
      action_name: actionName,
      ...properties,
    });
  }

  // 에러 추적
  error(errorMessage: string, error?: unknown): void {
    this.track('error', {
      error_message: errorMessage,
      error_details: error,
    });
  }

  // 성능 추적
  performance(metricName: string, value: number, properties?: Record<string, unknown>): void {
    this.track('performance', {
      metric_name: metricName,
      value,
      ...properties,
    });
  }

  // 이벤트 배치 전송
  private async flush(): Promise<void> {
    if (this.events.length === 0) return;

    const eventsToSend = [...this.events];
    this.events = [];

    try {
      if (this.config.endpoint) {
        await fetch(this.config.endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            events: eventsToSend,
            userProperties: this.userProperties,
          }),
        });
      } else {
        // 개발 환경에서는 콘솔에 출력
        console.log('Analytics Events:', eventsToSend);
      }
    } catch (error) {
      console.error('Failed to send analytics events:', error);
      // 실패한 이벤트를 다시 큐에 추가
      this.events.unshift(...eventsToSend);
    }
  }

  // 주기적 전송 타이머 시작
  private startFlushTimer(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }

    this.flushTimer = setInterval(() => {
      this.flush();
    }, this.config.flushInterval);
  }

  // 설정 업데이트
  updateConfig(config: Partial<AnalyticsConfig>): void {
    this.config = { ...this.config, ...config };
    this.startFlushTimer();
  }

  // 정리
  destroy(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    this.flush();
  }
}

// 기본 애널리틱스 인스턴스
export const analytics = new Analytics();

// 개발 환경에서는 로컬 스토리지에 저장
if (process.env.NODE_ENV === 'development') {
  analytics.updateConfig({
    endpoint: undefined, // 로컬에서만 콘솔 출력
  });
} 