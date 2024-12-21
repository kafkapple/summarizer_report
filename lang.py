import os
import json
from bs4 import BeautifulSoup
from pathlib import Path
from jsonschema import validate, ValidationError
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
# Langchain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.globals import set_verbose
from langchain.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import openai
import sys
import re
from utils import Utils
import pandas as pd

print(sys.path)
load_dotenv()
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# OpenAI API 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
from config import JsonFunction, Config

config = Config()
# JSON Function 정의
json_config = JsonFunction()
json_function = json_config.json_function_final

json_config.role
json_config.prompt


# 1. HTML 파일 파싱 및 데이터 추출
import re
def bert_summarize(text, sentence_count=3):
    from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
    # 모델과 토크나이저 로드
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # 요약 파이프라인 생성
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer(text, sentence_count=sentence_count)

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

nltk.download('stopwords')
nltk.download('punkt')

from doc_parse import render_html_from_json
# def compress_sentence_nltk(sentence):
#     """NLTK를 사용한 간단한 문장 압축"""
#     # 토큰화
#     tokens = word_tokenize(sentence)
    
#     # 한국어 불용어 목록 수동 설정
#     korean_stopwords = set([
#         '이', '그', '저', '것', '수', '등', '들', '및', '에서', '의', '를', '에', '가', '은', '는', '로', '과', '와', '도', '으로', '하고', '이다', '다', '고', '하', '한', '있', '되', '않', '없', '나', '사람', '주', '년', '월', '일', '시', '분', '초', '때', '곳', '앞', '뒤', '위', '아래', '안', '밖', '안쪽', '바깥쪽', '사이', '중간', '가운데', '옆', '왼쪽', '오른쪽', '위쪽', '아래쪽', '앞쪽', '뒤쪽'
#     ])
    
#     # 불용어 제거
#     compressed_tokens = [word for word in tokens if word.lower() not in korean_stopwords]
    
#     return ' '.join(compressed_tokens)

def compress_sentence_sumy(text, sentence_count=3):
    """Sumy 라이브러리를 사용한 문장 압축"""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    
    summary = summarizer(parser.document, sentences_count=sentence_count)
    return ' '.join(str(sentence) for sentence in summary)


def clean_text(text):
    # 알파벳, 숫자, 공백, 기본 구두점만 남기고 나머지 제거
    cleaned_text = re.sub(r'[^\w\s.,!?]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # 여러 공백을 하나의 공백으로
    return cleaned_text.strip()

def create_messages(text):
    return HumanMessage(content=f"다음의 학생 활동 내역을 바탕으로 분석하고 생활기록부 특기사항을 작성하세요.\n\n{json_config.prompt}\n\n{text}"), [SystemMessage(content=json_config.role), HumanMessage(content=f"다음은 원하는 결과물의 예시:\n\n{json_config.example}")]

def parse_filename(file_path: str) -> tuple[str, str]:
    """
    파일 이름에서 학번과 이름을 추출
    
    Args:
        file_path: HTML 파일 경로
    
    Returns:
        tuple[str, str]: (학번, 이름)
    """
    # 파일 이름만 추출 (확장자 제외)
    filename = Path(file_path).stem
    
    # '_'로 분리
    parts = filename.split('_')
    if len(parts) >= 2:
        student_id = parts[0]
        name = parts[1]
        return student_id, name
    return "", ""

def parse_html(file_path):
    """HTML 파일 파싱 및 데이터 추출"""
    # 파일 이름에서 학번과 이름 추출
    student_id, name = parse_filename(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    # student_info 설정
    student_info = {'student_id': student_id, 'name': name}
    
    # 테이블 내용 추출 (기존 로직 유지)
    tables = soup.find_all("table")
    extracted_data = []
    for table in tables:
        if "평가 기준" not in table.text:
            rows = table.find_all("tr")
            table_data = {}
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 2:
                    key = cells[0].text.strip()
                    value = cells[1].text.strip()
                    table_data[key] = value
            if table_data:
                extracted_data.append(table_data)

    return student_info, extracted_data

# 2. Vector DB 준비 (수학 교과 데이터 업로드)
def prepare_vector_db(pdf_paths, db_path="vector_db"):
    print(f"벡터 DB 준비 시작: {db_path}")
    
    import os
    import pickle
    
    processed_files_path = os.path.join(db_path, "processed_files.pkl")
    if os.path.exists(processed_files_path):
        with open(processed_files_path, "rb") as f:
            processed_files = pickle.load(f)
        print(f"기존 처리된 파일 수: {len(processed_files)}")
    else:
        processed_files = set()
        print("새로운 처리 파일 목록 생성")
    
    if os.path.exists(db_path):
        try:
            vector_store = FAISS.load_local(
                db_path, 
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True  # 안전한 환경에서만 사용
            )
            print(f"존 벡 DB 로드 완료: {len(processed_files)}개 파일")
        except Exception as e:
            print(f"벡터 DB 로드 실패, 새로 생성합니다: {e}")
            vector_store = None
    else:
        vector_store = None
        os.makedirs(db_path, exist_ok=True)
        print("새로운 벡터 DB 생성")
    
    # 새로운 파일 처리
    new_documents = []
    for pdf_path in pdf_paths:
        if str(pdf_path) not in processed_files:
            print(f"PDF 처리 중: {pdf_path}")
            try:
                loader = PyPDFLoader(str(pdf_path))
                new_documents.extend(loader.load())
                processed_files.add(str(pdf_path))
                print(f"PDF 처리 완료: {pdf_path}")
            except Exception as e:
                print(f"PDF 처리 실패: {pdf_path} - {e}")
    
    if new_documents:
        print(f"새로운 문서 분할 시작: {len(new_documents)}개")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(new_documents)
        print(f"문서 분할 완료: {len(texts)}개 청크")
        
        if vector_store is None:
            vector_store = FAISS.from_documents(texts, OpenAIEmbeddings())
        else:
            vector_store.add_documents(texts)
            
        # 저장
        vector_store.save_local(db_path)
        with open(processed_files_path, "wb") as f:
            pickle.dump(processed_files, f)
        print(f"벡터 DB 저장 완료: 새로 추가된 파일 {len(new_documents)}개")
    else:
        print("새로 추할 일이 없습니다")
    
    return vector_store

# 3. LangChain 기반 요약
class ChunkConfig:
    """청크 설정 관리 클래스"""
    # GPT-3.5 기준
    GPT35_CHUNK_SIZE = 1500  # 약 2K 토큰
    GPT35_OVERLAP = 150      # 10% 오버랩
    
    # GPT-4 기준
    GPT4_CHUNK_SIZE = 3000   # 약 4K 토큰
    GPT4_OVERLAP = 300       # 10% 오버랩

def summarize_content(
    data,
    json_function,
    llm,
    vector_store=None,
    chunk_size=None,
    overlap=None,
    validate_schema=True,
    k_relevant_docs=3,
    use_rag=False
):
    """
    데이터를 요약하는 통합 함수
    
    Args:
        chunk_size (int, optional): 청크 크기. 미지정시 모델에 따라 자동 설정
        overlap (int, optional): 오버랩 크기. 미지정시 청크 크기의 10%
    """
    # 모델에 따른 청크 크기 자동 설정
    if chunk_size is None:
        if "gpt-4" in llm.model_name:
            chunk_size = ChunkConfig.GPT4_CHUNK_SIZE
            overlap = ChunkConfig.GPT4_OVERLAP
        else:  # gpt-3.5
            chunk_size = ChunkConfig.GPT35_CHUNK_SIZE
            overlap = ChunkConfig.GPT35_OVERLAP
    
    # overlap이 따로 지정되지 않은 경우 청크 크기의 10%로 설정
    if overlap is None:
        overlap = int(chunk_size * 0.1)
        
    print(f"청크 설정 - 크기: {chunk_size}, 오버랩: {overlap}")
    
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False)

    # RAG 사용시 vector_store 확인
    if use_rag and vector_store is None:
        raise ValueError("RAG를 사용하려면 vector_store가 필요합니다.")

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(data)

    summaries = []
    for chunk in chunks:
        if use_rag:
            relevant_context = get_relevant_context(vector_store, chunk, k=k_relevant_docs)
            compressed_text = clean_text(relevant_context)
            compressed_text = compress_sentence_sumy(compressed_text)
            enhanced_prompt = f"""
                                분석할 데이터:
                                {clean_text(chunk)}
                                """
            prompt, messages = create_messages(enhanced_prompt)
        else:
            prompt, messages = create_messages(clean_text(chunk))
            
        try:
            all_messages = [*messages, prompt]  # 시스템 메시지, 예시, 그리고 실제 입력을 하나의 리스트로
            response = llm.invoke(all_messages)
            
            arguments = response.additional_kwargs["function_call"]["arguments"]
            summary = json.loads(arguments)
            summaries.append(summary)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"청크 처리 중 오류 발생: {e}")
            continue

    merged_summary = merge_summaries(summaries, json_function)

    if validate_schema:
        try:
            schema = json_function[0]["parameters"]
            validate(instance=merged_summary, schema=schema)
            print("JSON 검증 성공")
        except ValidationError as e:
            print(f"JSON 검증 실패: {e}")

    return merged_summary

def merge_summaries(summaries, json_function):
    """
    Merges individual JSON summaries into a single JSON object based on the specified JSON function schema.

    Args:
        summaries (list): List of JSON summary objects from each chunk.
        json_function (list): JSON function definition.

    Returns:
        dict: Merged JSON object following the JSON function schema.
    """
    schema = json_function[0]["parameters"]["properties"]
    merged_result = {}

    # Iterate over schema keys and merge
    for key in schema.keys():
        if schema[key]["type"] == "array":
            merged_result[key] = []
            for summary in summaries:
                if key in summary:
                    merged_result[key].extend(summary[key])
        elif schema[key]["type"] == "object":
            merged_result[key] = {}
            for summary in summaries:
                if key in summary:
                    for sub_key, value in summary[key].items():
                        merged_result[key][sub_key] = value 
        else:
            # For string or other types, concatenate or take the latest
            merged_result[key] = " ".join([summary[key] for summary in summaries if key in summary])

    return merged_result

# 4. JSON 저장
def save_as_json(result, student_info, output_dir):
    student_id = student_info["student_id"]
    name = student_info["name"]
    file_name = f"{student_id}_{name}.json"

    with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Saved: {file_name}")

# def save_as_csv(result, student_info, output_dir):
#     csv_path = os.path.join(output_dir, 'report.csv')
    
#     # 저장할 데이터 준비
#     data = {
#         'student_id': student_info['student_id'],
#         'name': student_info['name'],
#         'activity_info': result['activity_information']
#     }
    
#     # 기존 CSV 파일이 있는지 확인
#     if os.path.exists(csv_path):
#         df = pd.read_csv(csv_path)
#         df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
#     else:
#         df = pd.DataFrame([data])
    
#     # CSV 파일로 저장
#     df.to_csv(csv_path, index=False, encoding='utf-8-sig')

# OpenAI LangChain 설정
def setup_llm():
    print(f"LLM 설정: {config.TEMPERATURE}")
    return ChatOpenAI(
        model=config.GPT_MODEL,
        temperature=config.TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
        functions=json_config.json_function_final,  # 함수 정의 추가
        function_call={"name": json_config.json_function_final[0]["name"]}  # 함수 호출 설정
    )
# # 5. 통합 파이프라인 실행
# def process_pipeline(html_file, pdf_files=None, output_dir="output"):
#     os.makedirs(output_dir, exist_ok=True)

#     # HTML Parsing
#     student_info, extracted_data = parse_html(html_file)

#     # Vector DB 준비 (옵)
#     vector_store = prepare_vector_db(pdf_files) if pdf_files else None

#     # 데이터 요약 및 JSON 변환
#     result = summarize_data(
#         data={"student_info": student_info, "activities": extracted_data},
#         json_function=json_function,
#         llm=setup_llm(),
#         validate_schema=True
#     )

#     # JSON 저장
#     save_as_json(result, student_info, output_dir)

def get_relevant_context(vector_store, query, k=3):
    """
    력 쿼리와 련된 상위 k개의 문서를 색합니다.
    
    Args:
        vector_store (FAISS): 벡터 장소
        query (str): 검색 쿼리
        k (int): 검색할 문서 수
    
    Returns:
        str: 병합된 관련 컨텍스트
    """
    docs = vector_store.similarity_search(query, k=k)
    print(f"관련 컨텍스트 검색 완료: {len(docs)}개")
    
    return "\n\n".join([doc.page_content for doc in docs])

# 사용 예시:
def process_single_student(html_file, output_dir="output", vector_store=None, use_rag=False):
    print(f"\n=== 학생 데이터 처리 시작 ===")
    print(f"HTML 파일: {html_file}")
    print(f"출력 디렉토리: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # HTML 파싱
    print("\n1. HTML 파싱 시작")
    student_info, extracted_data = parse_html(html_file)
    print(f"학생 정보 추출: {student_info}")
    print(f"활동 데이터 수: {len(extracted_data)}")
    print(f"활동 데이터: {extracted_data}")
    
    # LLM 설정
    print("\n3. LLM 설정")
    set_verbose(True)
    llm = setup_llm()
    
    # 데이터 요약
    print("\n4. 데이터 요약 시작")

    result = summarize_content(
        data={"student_info": student_info, "activities": extracted_data},
        json_function=json_function,
        llm=llm,
        vector_store=vector_store,
        chunk_size=None,
        overlap=None,
        validate_schema=True,
        k_relevant_docs=3,
        use_rag=use_rag
    )
    # 결과 저장
    return result, student_info
# 실행

def process_students(student_list, output_dir, db_path, pdf_dir=None):
    all_results = []
    print(f"학생 목록: {student_list}")
    print(f"출력 디렉토리: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # RAG 사용 여부에 따라 vector_store 준비
    vector_store = None
    if pdf_dir:
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"발견된 PDF 파일: {len(pdf_files)}개")
        vector_store = prepare_vector_db(pdf_files, db_path=db_path)
        use_rag = True
    else:
        use_rag = False
    
    for student in student_list:
        result, student_info = process_single_student(student, output_dir=output_dir, vector_store=vector_store, use_rag=use_rag)
        print(f"결과: {result}")
        print(f"학생 정보: {student_info}")
        
        # 개별 JSON/HTML 저장
        render_html_from_json(
            json_data=result,
            student_id=student_info["student_id"],
            name=student_info["name"],
            output_dir=output_dir
        )
        
        # 결과 수집
        all_results.append({
            'student_id': student_info['student_id'],
            'name': student_info['name'],
            'activity_info': result['activity_information']
        })
    
    # CSV로 저장
    save_as_csv(all_results, output_dir, output_dir.stem)

def save_as_csv(results, output_dir, file_name):
    """
    결과를 CSV 파일로 저장 (학번 순 정렬)
    
    Args:
        results: 저장할 결과 리스트
        output_dir: 저장할 디렉토리 경로
        file_name: CSV 파일 이름 (확장자 제외)
    """
    csv_path = output_dir / f'{file_name}.csv'
    
    # DataFrame 생성
    df = pd.DataFrame(results)
    
    # student_id가 'unknown'이 아닌 행만 정수로 변환하여 정렬
    mask = df['student_id'].str.isdigit()
    
    # 숫자인 학번과 아닌 학번을 분리
    df_numeric = df[mask].copy()
    df_unknown = df[~mask].copy()
    
    # 숫자인 학번만 정렬
    if not df_numeric.empty:
        df_numeric['sort_id'] = df_numeric['student_id'].astype(int)
        df_numeric = df_numeric.sort_values('sort_id')
        df_numeric = df_numeric.drop('sort_id', axis=1)
    
    # 정렬된 숫자 학번과 unknown 학번을 다시 합침
    df_sorted = pd.concat([df_numeric, df_unknown])
    
    # CSV 파일로 저장
    df_sorted.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV 파일 저장 완료 (학번 순 정렬): {csv_path}")


if __name__ == "__main__":
    # 테스트용 코드
    base_dir = Path(__file__).parent
    target= '11반_보고서'
    html_dir = base_dir / 'data' / 'processed' /'html' / target
    print(f"HTML 디렉토리: {html_dir}")
    html_files = Utils.list_files(html_dir, 'html')
    print(f"HTML 파일 목록: {html_files}")
    #html_file = Path(__file__).parent / 'save' / 'html' / "10801_강윤아.html"
    data_dir = base_dir / 'data'
    pdf_dir = data_dir / 'db' / 'pdf'  # 디렉토리 경로
    db_path = data_dir / 'db'
    save_dir = base_dir / 'save'
    #for html_file in html_files:
    process_students(html_files, output_dir=save_dir / 'final' / target , db_path=db_path)
    
# pip install langchain beautifulsoup4 pinecone openai
# pip install pdfminer.six konlpy sumy nltk 
