mermaid

graph TB
    %% 클라이언트 레이어
    subgraph "Client Layer"
        WEB[Web Frontend]
        MOBILE[Mobile App]
        API_CLIENT[API Client]
    end

    %% API Gateway & Load Balancer
    subgraph "API Gateway"
        LB[Load Balancer]
        CORS[CORS Middleware]
    end

    %% FastAPI 애플리케이션 레이어
    subgraph "FastAPI Application"
        MAIN[main.py]
        
        subgraph "Routers"
            AUTH_R[auth_router]
            TOKEN_R[token_router]
            SCRIPT_R[script_router]
            ACTOR_R[actor_router]
            USER_AUDIO_R[user_audio_router]
            SCRIPT_AUDIO_R[script_audio_router]
            MYPAGE_R[mypage_router]
        end
        
        subgraph "Core Components"
            MODELS[models.py]
            SCHEMAS[schemas.py]
            DATABASE[database.py]
        end
    end

    %% 비동기 처리 레이어
    subgraph "Async Processing"
        BG_TASKS[BackgroundTasks]
        THREAD_POOL[ThreadPoolExecutor]
        HTTPX[httpx AsyncClient]
    end

    %% 외부 서비스 레이어
    subgraph "External Services"
        subgraph "AWS Services"
            S3[AWS S3<br/>Audio Storage]
        end
        
        subgraph "Analysis Servers"
            TOKEN_ANALYSIS[Token Analysis Server<br/>54.180.25.231:8000]
            SCRIPT_ANALYSIS[Script Analysis Server<br/>54.180.25.231:8001]
        end
    end

    %% 데이터베이스 레이어
    subgraph "Database Layer"
        POSTGRES[(PostgreSQL<br/>Main Database)]
        
        subgraph "Tables"
            USERS_T[users]
            TOKENS_T[tokens]
            SCRIPTS_T[scripts]
            WORDS_T[words]
            ACTORS_T[actors]
            ANALYSIS_T[analysis_results]
            BOOKMARKS_T[bookmarks]
        end
    end

    %% 배포 환경
    subgraph "Deployment"
        RAILWAY[Railway Platform]
        UVICORN[Uvicorn ASGI Server]
    end

    %% 연결 관계
    WEB --> LB
    MOBILE --> LB
    API_CLIENT --> LB
    
    LB --> CORS
    CORS --> MAIN
    
    MAIN --> AUTH_R
    MAIN --> TOKEN_R
    MAIN --> SCRIPT_R
    MAIN --> ACTOR_R
    MAIN --> USER_AUDIO_R
    MAIN --> SCRIPT_AUDIO_R
    MAIN --> MYPAGE_R
    
    USER_AUDIO_R --> BG_TASKS
    SCRIPT_AUDIO_R --> BG_TASKS
    
    BG_TASKS --> THREAD_POOL
    BG_TASKS --> HTTPX
    
    THREAD_POOL --> S3
    HTTPX --> TOKEN_ANALYSIS
    HTTPX --> SCRIPT_ANALYSIS
    
    TOKEN_ANALYSIS -.->|Webhook| USER_AUDIO_R
    SCRIPT_ANALYSIS -.->|Webhook| SCRIPT_AUDIO_R
    
    MODELS --> POSTGRES
    DATABASE --> POSTGRES
    
    POSTGRES --> USERS_T
    POSTGRES --> TOKENS_T
    POSTGRES --> SCRIPTS_T
    POSTGRES --> WORDS_T
    POSTGRES --> ACTORS_T
    POSTGRES --> ANALYSIS_T
    POSTGRES --> BOOKMARKS_T
    
    RAILWAY --> UVICORN
    UVICORN --> MAIN

    %% 스타일링
    classDef clientStyle fill:#e1f5fe
    classDef apiStyle fill:#f3e5f5
    classDef asyncStyle fill:#fff3e0
    classDef externalStyle fill:#e8f5e8
    classDef dbStyle fill:#fce4ec
    classDef deployStyle fill:#f1f8e9

    class WEB,MOBILE,API_CLIENT clientStyle
    class MAIN,AUTH_R,TOKEN_R,SCRIPT_R,ACTOR_R,USER_AUDIO_R,SCRIPT_AUDIO_R,MYPAGE_R,MODELS,SCHEMAS,DATABASE apiStyle
    class BG_TASKS,THREAD_POOL,HTTPX asyncStyle
    class S3,TOKEN_ANALYSIS,SCRIPT_ANALYSIS externalStyle
    class POSTGRES,USERS_T,TOKENS_T,SCRIPTS_T,WORDS_T,ACTORS_T,ANALYSIS_T,BOOKMARKS_T dbStyle
    class RAILWAY,UVICORN deployStyle