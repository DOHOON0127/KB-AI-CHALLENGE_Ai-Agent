import os, re, time, json
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.utils import embedding_functions
import nltk
from nltk.tokenize import sent_tokenize
import httpx
from dotenv import load_dotenv
import google.generativeai as genai
import logging
from functools import lru_cache
import numpy as np

# 로깅 설정
logger = logging.getLogger(__name__)

load_dotenv()

# NLTK 데이터 다운로드 (안전하게 처리)
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception as e:
    print(f"NLTK 데이터 다운로드 중 오류 (무시하고 계속): {e}")

# ---- LLM (Gemini) - 최소화된 호출 ----
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def _raw_llm_call(system: str, user: str) -> str:
    """원본 LLM 호출 함수"""
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system
    )
    try:
        resp = model.generate_content(
            user,
            generation_config={"temperature": 0.2}
        )
        return getattr(resp, "text", "") or ""
    except Exception as e:
        logger.error(f"🤖 LLM 호출 실패: {e}")
        return f"(생성 실패: {e})"

@lru_cache(maxsize=256)
def _cached_llm(system: str, user: str) -> str:
    """캐시된 LLM 호출"""
    return _raw_llm_call(system, user)

def llm_chat(system: str, user: str) -> str:
    """LLM 호출 (캐시 적용)"""
    logger.info(f"🤖 LLM 호출 시작 - 모델: {GEMINI_MODEL}")
    logger.info(f"🤖 LLM 시스템 지시: {system[:100]}...")
    logger.info(f"🤖 LLM 사용자 입력: {user[:100]}...")
    
    start_time = time.time()
    
    # 캐시에 태우기
    key_sys = (system or "").strip()
    key_user = (user or "").strip()
    result = _cached_llm(key_sys, key_user)
    
    llm_time = time.time() - start_time
    logger.info(f"🤖 LLM 호출 완료 - 시간: {llm_time:.2f}초, 결과 길이: {len(result)}자")
    logger.info(f"🤖 LLM 결과 미리보기: {result[:100]}...")
    
    return result

# ---- 로컬 임베딩 모델 (LLM 대체) ----
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    logger.info("✅ SentenceTransformer 로드 성공")
    
    # 전역 로드 (앱 시작시 1회 로드)
    _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # 선택: 더 정확한 재랭커 (GPU 있으면 더 빠름)
    try:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # 768-dim
        logger.info("✅ CrossEncoder 로드 성공")
    except Exception as e:
        _reranker = None
        logger.warning(f"⚠️ CrossEncoder 로드 실패: {e}")
        
except ImportError:
    logger.error("❌ SentenceTransformer 설치 필요: pip install sentence-transformers torch")
    _embedder = None
    _reranker = None

def _cos_sim(a, b):
    """코사인 유사도 계산"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def score_docs_by_similarity(query: str, docs: List[str]) -> List[float]:
    """문서 유사도 점수 계산 (LLM 대신 로컬 임베딩 사용)"""
    if not _embedder:
        logger.warning("⚠️ 임베딩 모델 없음, 간단한 키워드 기반 점수 사용")
        # 폴백: 간단한 키워드 기반 점수
        scores = []
        for doc in docs:
            q_terms = set(re.findall(r"[가-힣A-Za-z]+", query.lower()))
            d_terms = set(re.findall(r"[가-힣A-Za-z]+", doc.lower()))
            overlap = len(q_terms & d_terms) / (len(q_terms) + 1e-5)
            scores.append(min(1.0, overlap))
        return scores
    
    try:
        # 1) 임베딩 코사인 유사도
        q_emb = _embedder.encode([query], normalize_embeddings=True)[0]
        d_embs = _embedder.encode(docs, normalize_embeddings=True)
        cos_scores = [_cos_sim(q_emb, d_emb) for d_emb in d_embs]
        
        # 2) (선택) Cross-Encoder 재랭크 → 0~1로 스케일
        if _reranker:
            pairs = [[query, d] for d in docs]
            ce_scores = _reranker.predict(pairs)  # 대략 0~30 범위
            ce_scores = (ce_scores - np.min(ce_scores)) / (np.ptp(ce_scores) + 1e-9)
            # 하이브리드: 0.5*cos + 0.5*ce
            scores = [0.5*c + 0.5*e for c, e in zip(cos_scores, ce_scores)]
            return scores
        
        return cos_scores
        
    except Exception as e:
        logger.error(f"❌ 임베딩 계산 실패: {e}, 폴백 사용")
        # 폴백: 간단한 키워드 기반 점수
        scores = []
        for doc in docs:
            q_terms = set(re.findall(r"[가-힣A-Za-z]+", query.lower()))
            d_terms = set(re.findall(r"[가-힣A-Za-z]+", doc.lower()))
            overlap = len(q_terms & d_terms) / (len(q_terms) + 1e-5)
            scores.append(min(1.0, overlap))
        return scores

# ---- Local Retriever (Chroma) ----
DB_DIR = "chroma_db"
_chroma_client = chromadb.PersistentClient(path=DB_DIR)
_collection = _chroma_client.get_or_create_collection(
    name="fin_consumer_corpus",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
)

def local_retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """벡터DB 검색 (k를 8→5로 축소)"""
    logger.info(f"🔍 벡터DB 검색 시작 - 쿼리: {query[:50]}..., k: {k}")
    
    start_time = time.time()
    res = _collection.query(query_texts=[query], n_results=k)
    search_time = time.time() - start_time
    
    docs = []
    for i in range(len(res["ids"][0])):
        docs.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i]
        })
    
    logger.info(f"🔍 벡터DB 검색 완료 - 시간: {search_time:.3f}초, 결과: {len(docs)}개")
    
    # 검색된 문서 요약 로깅
    for i, doc in enumerate(docs[:3]):
        title = doc.get('meta', {}).get('title', '제목없음')
        logger.info(f"🔍 벡터DB 결과 {i+1}: {title}")
    
    return docs

# ---- Retrieval Evaluator (LLM 제거, 로컬 임베딩 사용) ----
def evaluate_retrieval(query: str, docs: List[Dict[str, Any]], tau_low=0.35, tau_high=0.75):
    """검색 결과 평가 (LLM 대신 로컬 임베딩 사용)"""
    logger.info(f"📊 검색 결과 평가 시작 - 문서 {len(docs)}개, 임계값: low={tau_low}, high={tau_high}")
    
    if not docs:
        logger.info("📊 평가 완료 - 문서 없음")
        return "incorrect", []
    
    start_time = time.time()
    
    # LLM 대신 로컬 임베딩으로 점수 계산
    texts = [d["text"] for d in docs]
    scores = score_docs_by_similarity(query, texts)
    
    eval_time = time.time() - start_time
    
    has_high = any(s >= tau_high for s in scores)
    all_low  = all(s < tau_low for s in scores)
    action = "correct" if has_high else ("incorrect" if all_low else "ambiguous")
    
    logger.info(f"📊 평가 완료 - 시간: {eval_time:.3f}초, 액션: {action}")
    logger.info(f"📊 스코어 분포: 최고={max(scores) if scores else 'N/A'}, 최저={min(scores) if scores else 'N/A'}")
    logger.info(f"📊 임계값 판정: high(≥{tau_high})={has_high}, low(<{tau_low})={all_low}")
    
    return action, scores

# ---- Knowledge Refinement (LLM 제거, 로컬 임베딩 사용) ----
def refine_internal_knowledge(query: str, docs: List[Dict[str, Any]], scores: List[float], 
                             keep_topk=3, strip_tau=0.45, max_strips=20):
    """지식 정제 (LLM 대신 로컬 임베딩 사용)"""
    logger.info(f"🔄 지식 정제 시작 - 상위 {keep_topk}개 문서, 최대 {max_strips}개 문장")
    
    # 상위 문서 3개만 (keep_topk 축소)
    paired = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:keep_topk]
    
    candidates = []
    for d, _ in paired:
        try:
            sentences = sent_tokenize(d["text"])
            for sent in sentences:
                s = sent.strip()
                if 8 <= len(s) <= 300:  # 너무 짧거나 긴 문장 제외
                    candidates.append((s, d))
        except Exception as e:
            # NLTK 토크나이저 실패시 간단한 문장 분리
            logger.warning(f"문장 토큰화 실패, 간단 분리 사용: {e}")
            simple_sentences = [s.strip() for s in d["text"].split('.') if s.strip()]
            for sent in simple_sentences:
                if 8 <= len(sent) <= 300:
                    candidates.append((sent, d))
    
    if not candidates:
        logger.warning("🔄 정제할 문장 후보 없음")
        return "", []
    
    # 문장별 유사도 점수 (LLM 대신 로컬 임베딩)
    sent_texts = [c[0] for c in candidates]
    sent_scores = score_docs_by_similarity(query, sent_texts)
    
    # 상위 점수 문장 선별
    paired_sentences = list(zip(sent_scores, candidates))
    kept = [s for sc, (s, _) in sorted(paired_sentences, key=lambda x: x[0], reverse=True)
            if sc >= strip_tau][:max_strips]
    
    # citation: 상위 문서 메타에서
    citations = []
    seen = set()
    for d, sc in paired:
        url = d["meta"].get("url", "")
        if url and url not in seen:
            seen.add(url)
            citations.append({
                "title": d["meta"].get("title", "제목없음"), 
                "url": url, 
                "date": d["meta"].get("date", "")
            })
    
    result_text = "\n".join(kept)
    logger.info(f"🔄 지식 정제 완료: {len(result_text)}자, 인용 {len(citations)}개")
    
    return result_text, citations

# ---- Web Search ----
def web_search_and_scrape(query: str, topn=3) -> List[Dict[str, Any]]:
    """웹 검색 및 스크래핑"""
    logger.info(f"🌐 웹 검색 시작 - topn: {topn}")
    
    out = []
    cse_id = os.getenv("GOOGLE_CSE_ID")
    gkey   = os.getenv("GOOGLE_API_KEY")
    serpkey= os.getenv("SERPAPI_API_KEY")
    
    try:
        if cse_id and gkey:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {"key": gkey, "cx": cse_id, "q": query, "num": topn}
            r = httpx.get(url, params=params, timeout=20)
            r.raise_for_status()
            items = r.json().get("items", [])
            for it in items:
                text = fetch_and_extract(it["link"])
                if text:
                    out.append({"id": it["link"], "text": text, "meta": {"title": it.get("title",""), "url": it["link"], "date": ""}})
        elif serpkey:
            url = "https://serpapi.com/search"
            params = {"engine":"google","q":query,"api_key":serpkey}
            r = httpx.get(url, params=params, timeout=20)
            r.raise_for_status()
            for it in (r.json().get("organic_results") or [])[:topn]:
                link = it.get("link")
                text = fetch_and_extract(link)
                if text:
                    out.append({"id": link, "text": text, "meta": {"title": it.get("title",""), "url": link, "date": ""}})
        
        logger.info(f"🌐 웹 검색 완료: {len(out)}개 문서")
        return out
        
    except Exception as e:
        logger.error(f"🌐 웹 검색 실패: {e}")
        return []

def fetch_and_extract(url: str) -> str:
    """URL에서 텍스트 추출"""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url, timeout=10)
        if not downloaded:
            return ""
        return trafilatura.extract(downloaded) or ""
    except Exception:
        return ""

# ---- 최종 답안 생성 (LLM 1회만 호출) ----
ANSWER_SYSTEM = """너는 금융 소비자 권익 보호 도우미다. 
법률 자문이 아닌 정보 제공임을 고지하고, 단계별 절차/필요 서류/접수 경로/근거 출처를 구조화해라.
한국어로 간결하게."""

def generate_structured_answer(query: str, internal_text: str, external_text: str, citations: List[Dict[str,str]]) -> str:
    """최종 답변 생성 (LLM 1회만 호출)"""
    logger.info(f"🎯 최종 답변 생성 시작 - 내부: {len(internal_text)}자, 외부: {len(external_text)}자, 인용: {len(citations)}개")
    
    cite_str = "\n".join([f"- {c['title']} ({c['date']}) {c['url']}" for c in citations])
    user = f"""[사용자 상황]
{query}

[내부 지식(정제)]
{internal_text[:3000]}

[외부 지식(보강)]
{external_text[:3000]}

[출처 후보]
{cite_str}

위 정보를 바탕으로 아래 형식으로 작성:
1) 한 줄 요약
2) 단계별 절차 (번호 목록)
3) 필요 서류
4) 접수 채널/링크
5) 근거(문서명/개정일/URL)
"""
    
    # LLM 호출 (1회만!)
    result = llm_chat(ANSWER_SYSTEM, user)
    
    logger.info(f"🎯 최종 답변 생성 완료: {len(result)}자")
    return result