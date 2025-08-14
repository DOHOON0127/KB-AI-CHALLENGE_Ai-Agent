import streamlit as st
from agent_graph import app
import logging
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="금융 소비자 권익 보호 AI", layout="wide")

st.title("금융 소비자 권익 보호 AI 에이전트 (MVP)")
st.caption("본 서비스는 정보 제공 목적이며 법률 자문이 아닙니다.")

with st.sidebar:
    st.header("질문 예시")
    st.write("- 해외 결제가 떴는데 사용한 적이 없어요. 어떻게 하나요?")
    st.write("- 대출 중도상환 수수료가 계약서보다 많이 나왔어요.")
    st.write("- 보험금 지급 거절을 당했는데 대응 절차가 궁금해요.")

query = st.text_area("당신의 금융 분쟁 상황/질문을 적어주세요", height=140,
                     placeholder="예) 해외 미사용 카드 결제 발생. 카드사 이의제기와 금감원 분쟁조정 절차가 궁금합니다.")

go = st.button("안내 받기", type="primary")

if go and query.strip():
    # 🚀 사용자 입력 시작 로깅
    start_time = time.time()
    logger.info(f"🚀 사용자 입력 시작: {query[:50]}...")
    st.info(f"🚀 사용자 입력: {query}")
    
    with st.spinner("분석 중..."):
        # 📊 상태 초기화
        state = {"query": query}
        logger.info(f"📊 초기 상태 설정: {state}")
        
        # 🔄 에이전트 실행 (캐시 적용)
        logger.info("🔄 에이전트 실행 시작...")
        
        # Streamlit 캐시 적용
        @st.cache_data(ttl=3600)
        def run_graph(q):
            return app.invoke({"query": q})
        
        result = run_graph(query)
        
        # ⏱️ 실행 시간 계산
        execution_time = time.time() - start_time
        logger.info(f"⏱️ 총 실행 시간: {execution_time:.2f}초")
        
        # 📋 결과 로깅
        logger.info(f"📋 최종 결과: {result.get('answer', '')[:100]}...")
        logger.info(f"📋 액션: {result.get('action', 'N/A')}")
        logger.info(f"📋 스코어: {result.get('scores', [])}")
        logger.info(f"📋 인용: {len(result.get('citations', []))}개")
    
    # 🎯 결과 표시
    st.subheader("결과")
    st.write(result["answer"])
    
    # 📊 실행 정보 표시
    with st.expander("🔍 실행 정보 (개발용)"):
        st.metric("실행 시간", f"{execution_time:.2f}초")
        st.metric("액션", result.get("action", "N/A"))
        st.metric("스코어 개수", len(result.get("scores", [])))
        st.metric("인용 개수", len(result.get("citations", [])))
        
        # 로그 출력
        st.code(f"""
🚀 사용자 입력: {query}
📊 초기 상태: {state}
🔄 에이전트 실행 완료
⏱️ 실행 시간: {execution_time:.2f}초
📋 최종 결과: {result.get('answer', '')[:200]}...
        """)
    
    with st.expander("근거/출처 보기"):
        for c in result.get("citations", []):
            st.markdown(f"- [{c.get('title','출처')}]({c.get('url','')}) {c.get('date','')}")
    
    with st.expander("디버그(개발용)"):
        st.json({
            "action": result.get("action"),
            "scores": result.get("scores"),
            "execution_time": execution_time,
            "query": query,
            "result_keys": list(result.keys())
        })
else:
    st.info("좌측 예시를 참고해 질문을 입력하고 '안내 받기'를 눌러보세요.")