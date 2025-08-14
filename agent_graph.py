from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from tools import local_retrieve, evaluate_retrieval, refine_internal_knowledge, web_search_and_scrape, generate_structured_answer
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

class State(TypedDict, total=False):
    messages: List[Any]
    query: str
    docs: List[Dict[str, Any]]
    action: str               # "correct" | "incorrect" | "ambiguous"
    scores: List[float] 
    internal_text: str
    external_text: str
    citations: List[Dict[str,str]]
    answer: str
    source: str              # "vector_db" | "llm_generated" | "hybrid"

def node_intake(state: State) -> State:
    """📥 사용자 입력 처리 노드"""
    q = state["query"]
    logger.info(f"📥 노드: intake - 사용자 질문: {q[:50]}...")
    
    state["messages"] = [HumanMessage(content=q)]
    logger.info(f"📥 노드: intake - 메시지 생성 완료")
    
    return state

def node_retrieve(state: State) -> State:
    """🔍 벡터DB 검색 노드"""
    logger.info(f"🔍 노드: retrieve - 벡터DB 검색 시작")
    
    state["docs"] = local_retrieve(state["query"], k=5)  # k를 5로 축소
    logger.info(f"🔍 노드: retrieve - 검색 완료, 문서 {len(state['docs'])}개 발견")
    
    # 검색된 문서 요약 로깅
    for i, doc in enumerate(state["docs"][:3]):
        logger.info(f"🔍 노드: retrieve - 문서 {i+1}: {doc.get('meta', {}).get('title', '제목없음')}")
    
    return state

def node_evaluate(state: State) -> State:
    """📊 검색 결과 평가 노드 (로컬 임베딩 사용)"""
    logger.info(f"📊 노드: evaluate - 검색 결과 평가 시작")
    
    action, scores = evaluate_retrieval(state["query"], state["docs"])
    state["action"] = action
    state["scores"] = scores
    
    logger.info(f"📊 노드: evaluate - 평가 완료: action={action}, scores={scores}")
    logger.info(f"📊 노드: evaluate - 상위 스코어: {max(scores) if scores else 'N/A'}")
    
    return state

def node_corrective_rag(state: State) -> State:
    """🔄 Corrective RAG 분기 노드"""
    logger.info(f"🔄 노드: corrective_rag - Corrective RAG 분기 시작")
    
    # 벡터DB에 좋은 결과가 있는 경우
    if state["action"] == "correct":
        logger.info(f"🔄 노드: corrective_rag - 벡터DB 결과 우수, 즉시 답변 생성")
        
        # 내부 지식 정제 (LLM 없이 로컬 임베딩)
        internal_text, citations = refine_internal_knowledge(state["query"], state["docs"], state["scores"])
        
        state["internal_text"] = internal_text
        state["external_text"] = ""
        state["citations"] = citations
        state["source"] = "vector_db"
        
        logger.info(f"🔄 노드: corrective_rag - 내부 지식 정제 완료: {len(internal_text)}자, 인용 {len(citations)}개")
        
    # 벡터DB 결과가 부족한 경우
    else:
        logger.info(f"🔄 노드: corrective_rag - 벡터DB 결과 부족, LLM 보강 시작")
        
        # 내부 지식 정제
        internal_text, citations = refine_internal_knowledge(state["query"], state["docs"], state["scores"])
        
        # 웹 검색으로 외부 지식 보강
        web_docs = web_search_and_scrape(state["query"], topn=3)
        external_text = "\n".join([(d["text"] or "")[:1000] for d in web_docs])
        
        # 웹 검색 결과를 인용에 추가
        for d in web_docs[:3]:
            citations.append({"title": d["meta"].get("title","웹자료"), "url": d["meta"]["url"], "date": ""})
        
        state["internal_text"] = internal_text
        state["external_text"] = external_text
        state["citations"] = citations
        state["source"] = "llm_generated"
        
        logger.info(f"🔄 노드: corrective_rag - 하이브리드 완료: 내부 {len(internal_text)}자, 외부 {len(external_text)}자, 인용 {len(citations)}개")
    
    return state

def node_generate(state: State) -> State:
    """🎯 최종 답변 생성 노드 (LLM 1회만 호출)"""
    logger.info(f"🎯 노드: generate - 최종 답변 생성 시작")
    
    state["answer"] = generate_structured_answer(
        state["query"], state.get("internal_text",""), state.get("external_text",""), state.get("citations",[])
    )
    
    logger.info(f"🎯 노드: generate - 답변 생성 완료: {len(state['answer'])}자")
    logger.info(f"🎯 노드: generate - 답변 미리보기: {state['answer'][:100]}...")
    logger.info(f"🎯 노드: generate - 소스: {state.get('source', 'unknown')}")
    
    return state

# 그래프 구성 (Corrective RAG 구조)
graph = StateGraph(State)
graph.add_node("intake", node_intake)
graph.add_node("retrieve", node_retrieve)
graph.add_node("evaluate", node_evaluate)
graph.add_node("corrective_rag", node_corrective_rag)  # 기존 refine_or_web 대신
graph.add_node("generate", node_generate)

graph.set_entry_point("intake")
graph.add_edge("intake", "retrieve")
graph.add_edge("retrieve", "evaluate")
graph.add_edge("evaluate", "corrective_rag")  # 평가 결과에 따라 분기
graph.add_edge("corrective_rag", "generate")
graph.add_edge("generate", END)

app = graph.compile()