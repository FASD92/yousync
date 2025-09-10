// 로그 레벨 정의
export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
}

// 로그 설정
interface LoggerConfig {
  level: LogLevel;
  enableConsole: boolean;
  enableRemote: boolean;
  remoteEndpoint?: string;
}

// 기본 설정
const defaultConfig: LoggerConfig = {
  level: LogLevel.INFO,
  enableConsole: true,
  enableRemote: false,
};

class Logger {
  private config: LoggerConfig;

  constructor(config: Partial<LoggerConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
  }

  // 디버그 로그
  debug(message: string, data?: unknown): void {
    this.log(LogLevel.DEBUG, message, data);
  }

  // 정보 로그
  info(message: string, data?: unknown): void {
    this.log(LogLevel.INFO, message, data);
  }

  // 경고 로그
  warn(message: string, data?: unknown): void {
    this.log(LogLevel.WARN, message, data);
  }

  // 에러 로그
  error(message: string, error?: unknown): void {
    this.log(LogLevel.ERROR, message, error);
  }

  // 실제 로그 처리
  private log(level: LogLevel, message: string, data?: unknown): void {
    if (level < this.config.level) return;

    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level: LogLevel[level],
      message,
      data,
    };

    // 콘솔 로그
    if (this.config.enableConsole) {
      this.logToConsole(level, logEntry);
    }

    // 원격 로그
    if (this.config.enableRemote && this.config.remoteEndpoint) {
      this.logToRemote(logEntry);
    }
  }

  // 콘솔에 로그 출력
  private logToConsole(level: LogLevel, logEntry: unknown): void {
    const { message, data } = logEntry as { message: string; data?: unknown };
    
    switch (level) {
      case LogLevel.DEBUG:
        console.debug(message, data);
        break;
      case LogLevel.INFO:
        console.info(message, data);
        break;
      case LogLevel.WARN:
        console.warn(message, data);
        break;
      case LogLevel.ERROR:
        console.error(message, data);
        break;
    }
  }

  // 원격 서버에 로그 전송
  private async logToRemote(logEntry: unknown): Promise<void> {
    try {
      await fetch(this.config.remoteEndpoint!, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(logEntry),
      });
    } catch (error) {
      console.error('Failed to send log to remote server:', error);
    }
  }

  // 설정 업데이트
  updateConfig(config: Partial<LoggerConfig>): void {
    this.config = { ...this.config, ...config };
  }
}

// 기본 로거 인스턴스
export const logger = new Logger();

// 개발 환경에서는 디버그 레벨로 설정
if (process.env.NODE_ENV === 'development') {
  logger.updateConfig({ level: LogLevel.DEBUG });
} 