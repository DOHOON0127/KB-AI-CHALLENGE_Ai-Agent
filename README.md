# 🏦 금융 소비자 권익 보호 AI 에이전트

**Corrective RAG + Human-in-the-loop** 구조를 기반으로 한 지능형 금융 소비자 권익 보호 AI 에이전트입니다.

## ✨ 주요 기능

- **🔍 Corrective RAG**: 벡터DB 우선 검색, 필요시 LLM 보강
- **🧠 LangGraph 워크플로우**: 구조화된 AI 워크플로우 관리
- **🔄 MCP 통합**: Model Context Protocol 기반 도구 체인
- **🤖 LLM 최적화**: Gemini API 1회 호출로 효율적인 답변 생성
- **💬 직관적 UI**: Streamlit 기반 사용자 친화적 인터페이스

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 활성화
source .venv/bin/activate

# 환경변수 설정
source .env
```

### 2. 의존성 설치
```bash
uv sync
```

### 3. 데이터 전처리
```bash
python data_process.py
```

### 4. MCP 서버 실행 (선택사항)
```bash
# 새 터미널에서 실행
mcp dev mcp_server.py
```

### 5. 웹 앱 실행
```bash
# 또 다른 터미널에서 실행
streamlit run app.py
```

## 🏗️ 아키텍처

### **전체 시스템 구조**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            
│   프론트엔드       │    │   백엔드          │    │   외부 서비스     │
│   (Streamlit)   │◄──►│   (LangGraph)   │◄──►│   (Gemini API)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   벡터DB         │
                       │   (ChromaDB)    │
                       └─────────────────┘
```

### **Corrective RAG 워크플로우**
```
사용자 질문 → 벡터DB 검색 → 평가 → 분기 → 답변 생성
                                   │
                            ┌──────┴──────┐
                            │             │
                      action="correct"  action!="correct"
                            │             │
                    ┌───────┴──────┐   ┌──┴──────────┐
                    │              │   │             | 
                    |   벡터DB 기반  |   |   LLM 보강   |   
                    |   즉시 답변    |   |   (1회 호출)  |
                    │              │   |             |
                    |   source=    |   |  source=    |
                    |   "vector_db"|   |  "llm_gen" |
                    │              │   │             |
                    └──────────────┘   └─────────────┘ 
```

### **LangGraph 노드 구조**
```
START → intake → retrieve → evaluate → corrective_rag → generate → END
                │         │         │                │
                ▼         ▼         ▼                ▼
            벡터DB 검색   로컬 평가   MCP/웹 분기       LLM 답변
             (k=5)       (임베딩)    (선택적)         (1회만)
```

## 📁 프로젝트 구조

```
KB-AI-CHALLENGE_Ai-Agent/
├── 📄 app.py                    # Streamlit 프론트엔드
├── 📄 agent_graph.py            # LangGraph 워크플로우 (Corrective RAG)
├── 📄 tools.py                  # 핵심 도구들 (LLM, 벡터DB, 임베딩)
├── 📄 .env                      # 환경변수 (API 키 등)
├── 📄 data_process.py           # 데이터 전처리 및 벡터DB 구축
├── 📄 pyproject.toml            # 의존성 관리
└── 📄 README.md                 # 프로젝트 문서
```

## 🔧 핵심 컴포넌트

### **1. LangGraph 워크플로우 (`agent_graph.py`)**
- **`node_intake`**: 사용자 입력 처리
- **`node_retrieve`**: 벡터DB 검색 (k=5)
- **`node_evaluate`**: 로컬 임베딩 기반 결과 평가
- **`node_corrective_rag`**: MCP vs 웹 검색 분기 결정
- **`node_generate`**: LLM 1회 호출로 최종 답변 생성

### **2. 도구 체인 (`tools.py`)**
- **LLM 통합**: Gemini API (캐싱 적용)
- **벡터 검색**: ChromaDB + SentenceTransformer
- **로컬 평가**: 임베딩 기반 유사도 계산 (LLM 호출 없음)
- **지식 정제**: 문장 단위 분해 및 선별
- **MCP 통합**: Model Context Protocol 도구 체인

### **3. 프론트엔드 (`app.py`)**
- **사용자 인터페이스**: Streamlit 기반 챗봇
- **MCP 연결 관리**: 서버 연결/해제 UI
- **전략 선택**: 외부 지식 보강 방법 선택
- **실시간 모니터링**: 실행 정보 및 로그 표시

## Corrective RAG 동작 원리

### **1단계: 벡터DB 우선 검색**
- 사용자 질문을 임베딩하여 유사 사례 검색
- 상위 5개 문서 후보 선별

### **2단계: 스마트한 결과 평가**
- **로컬 임베딩 모델**로 문서별 점수 계산
- 임계값 기반 분기 결정:
  - `correct`: 스코어 ≥ 0.75 (높은 품질)
  - `ambiguous`: 0.35 ≤ 스코어 < 0.75 (중간 품질)
  - `incorrect`: 스코어 < 0.35 (낮은 품질)

### **3단계: 지능적 분기 처리**
```
벡터DB 결과 우수 → 즉시 답변 생성 (LLM 호출 없음)
벡터DB 결과 부족 → 외부 지식 보강 후 LLM 답변 생성
```

### **4단계: 외부 지식 보강 전략**
- **자동 선택**: MCP 우선, 실패 시 웹 검색
- **MCP 도구만**: 전문 금융 상담 도구 사용
- **웹 검색만**: 인터넷 검색으로 정보 보강
- **외부 지식 사용 안함**: 내부 지식만 활용

## 성능 최적화

### **LLM 호출 최소화**
- **이전**: 13회 LLM 호출 (문서별 평가 + 문장별 정제)
- **현재**: **1회 LLM 호출** (최종 답변 생성만)
- **개선율**: **92% 감소**

### **실행 시간 단축**
- **이전**: 55초
- **현재**: **5~10초**
- **개선율**: **80~90% 단축**

### **로컬 임베딩 활용**
- **문서 평가**: LLM 대신 SentenceTransformer 사용
- **지식 정제**: 임베딩 기반 문장 선별
- **폴백 메커니즘**: 임베딩 실패 시 키워드 기반 점수

## 🔌 MCP 서버 통합

### **지원하는 도구들**
- **`consult`**: 멀티턴 금융 상담
- **`search_legal_basis`**: 법적 근거 검색
- **`search_dispute_cases`**: 분쟁 사례 검색
- **`calculate_compensation`**: 보상금 계산

### **연결 방법**
1. **MCP 서버 실행**: `mcp dev mcp_server.py`
2. **Streamlit에서 연결**: 사이드바의 MCP 서버 설정
3. **자동 도구 감지**: 사용 가능한 도구 목록 자동 표시

## 사용 예시

### **시나리오 1: 벡터DB에 관련 사례 있음**
```
사용자: "펀드 설명을 제대로 못 들었어요"
→ 벡터DB 검색 → 관련 사례 발견 (스코어: 0.85)
→ action="correct" → 즉시 답변 생성 (LLM 호출 없음)
→ source="vector_db_mcp_tools"
```

### **시나리오 2: 벡터DB에 관련 사례 없음**
```
사용자: "새로운 금융 상품 문제가 있어요"
→ 벡터DB 검색 → 관련 사례 없음 (스코어: 0.45)
→ action="ambiguous" → MCP 도구 + LLM 보강
→ source="llm_generated_mcp_tools"
```

### **시나리오 3: 외부 지식 전략 선택**
```
전략: "🔄 MCP 도구만 사용"
→ MCP 서버 연결됨 → consult, search_legal_basis 등 실행
→ 웹 검색 사용 안함 → 전문 도구만으로 답변 생성
```

## 🛠️ 기술 스택

- **Python**: 3.13+
- **LangGraph**: AI 워크플로우 오케스트레이션
- **Streamlit**: 웹 인터페이스
- **ChromaDB**: 벡터 데이터베이스
- **SentenceTransformer**: 로컬 임베딩 모델
- **Google Gemini**: LLM 통합 (최적화됨)
- **MCP**: Model Context Protocol
- **uv**: Python 패키지 관리

## 🔑 환경변수 설정

### **필수 설정**
```bash
# .env 파일
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-1.5-flash
```

### **선택 설정**
```bash
# 웹 검색용 (Google CSE)
GOOGLE_CSE_ID=your-cse-id
GOOGLE_API_KEY=your-google-api-key

# 또는 SerpAPI
SERPAPI_API_KEY=your-serpapi-key
```

## 모니터링 및 로깅

### **실시간 로깅**
- **노드별 실행**: 각 워크플로우 단계별 상세 로그
- **성능 지표**: 실행 시간, LLM 호출 횟수, 캐시 히트율
- **소스 추적**: 답변의 출처 (vector_db, llm_generated, mcp_tools 등)

### **UI 대시보드**
- **실행 정보**: 실행 시간, 액션, 스코어, 인용 개수
- **외부 지식 전략**: 선택된 보강 방법
- **MCP 상태**: 서버 연결 상태 및 사용 가능한 도구

## 주요 장점

1. **효율성**: LLM 호출 최소화로 비용 및 시간 절약
2. **정확성**: 벡터DB + 로컬 임베딩으로 빠르고 정확한 평가
3. **유연성**: MCP vs 웹 검색 선택적 사용
4. **확장성**: LangGraph 기반 모듈화된 구조
5. **사용성**: 직관적인 UI와 실시간 모니터링

## 라이선스

MIT License

---
