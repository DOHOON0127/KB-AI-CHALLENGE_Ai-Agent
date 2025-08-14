import json
import re
import os
from sentence_transformers import SentenceTransformer
import chromadb
from typing import Dict

# 1. 텍스트 파싱 함수 (형식에 맞게)
def parse_case(text: str, case_id: str) -> Dict:
    """금감원 사례 형식에 맞춰 파싱"""
    
    # 제목 추출 (첫 줄)
    lines = text.strip().split('\n')
    title = lines[0].strip() if lines else "제목 없음"
    
    # 각 섹션 추출
    sections = {}
    section_names = ['민원내용', '쟁점', '처리결과', '소비자 유의사항']
    
    for section in section_names:
        pattern = f'▣ {section}(.*?)(?=▣|$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[section] = match.group(1).strip()
    
    # 카테고리 자동 분류
    category = "기타"
    if '펀드' in text:
        category = "펀드"
    elif '보험' in text:
        category = "보험"
    elif '대출' in text:
        category = "대출"
    elif '카드' in text:
        category = "카드"
    
    # 판정 결과 분석
    decision = "미확정"
    result_text = sections.get('처리결과', '')
    if '보기 어려움' in result_text or '인정하기 어려움' in result_text:
        decision = "소비자 패소"
    elif '인정됨' in result_text or '타당함' in result_text:
        decision = "소비자 승소"
    
    return {
        "id": case_id,
        "title": title,
        "category": category,
        "complaint": sections.get('민원내용', ''),
        "issue": sections.get('쟁점', ''),
        "result": sections.get('처리결과', ''),
        "advice": sections.get('소비자 유의사항', ''),
        "decision": decision,
        "full_text": text  # 전체 텍스트도 저장
    }

# 2. 모든 파일 처리 함수
def process_all_files(folder_path: str):
    """폴더 내 모든 txt 파일 처리"""
    
    # 임베딩 모델 & ChromaDB 설정
    print("🔄 모델 로딩 중...")
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 컬렉션 생성/가져오기
    try:
        collection = client.create_collection("cases")
        print("✅ 새 컬렉션 생성")
    except:
        # 기존 컬렉션 삭제 후 재생성 (초기화)
        client.delete_collection("cases")
        collection = client.create_collection("cases")
        print("♻️ 컬렉션 초기화")
    
    # 모든 txt 파일 처리
    all_cases = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    print(f"📂 {len(files)}개 파일 발견")
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        case_id = filename.replace('.txt', '')
        
        # 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 파싱
        case = parse_case(text, case_id)
        all_cases.append(case)
        
        # 임베딩 생성
        embedding = model.encode(case["full_text"]).tolist()
        
        # ChromaDB에 저장
        collection.add(
            ids=[case["id"]],
            embeddings=[embedding],
            documents=[case["full_text"]],
            metadatas={
                "title": case["title"],
                "category": case["category"],
                "decision": case["decision"]
            }
        )
        
        print(f"✅ {case_id} 처리 완료 - {case['title'][:30]}...")
    
    # JSON 백업 저장
    with open("all_cases.json", "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 완료! 총 {len(all_cases)}개 사례 처리됨")
    print("💾 all_cases.json 파일 생성됨")
    print("🗄️ ChromaDB에 벡터 저장됨")
    
    return all_cases

# 3. 검색 테스트 함수
def search_cases(query: str, n_results: int = 3):
    """유사 사례 검색"""
    
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("cases")
    
    # 검색
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    print(f"\n🔍 '{query}' 검색 결과:")
    print("-" * 50)
    
    for i in range(len(results['ids'][0])):
        print(f"\n📌 {i+1}. {results['ids'][0][i]}")
        print(f"   제목: {results['metadatas'][0][i]['title']}")
        print(f"   카테고리: {results['metadatas'][0][i]['category']}")
        print(f"   판정: {results['metadatas'][0][i]['decision']}")
        print(f"   내용: {results['documents'][0][i][:100]}...")
    
    return results

# 4. 실행 코드
if __name__ == "__main__":
    # 모든 파일 처리 (파일들이 있는 폴더 경로 지정)
    folder_path = "raw_data"  # raw_data 폴더에서 txt 파일 검색
    process_all_files(folder_path)
    
    # 검색 테스트
    print("\n" + "="*50)
    search_cases("펀드 설명을 제대로 못 들었어요")
    search_cases("자필 서명 안했는데")