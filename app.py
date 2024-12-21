import streamlit as st
import os
from pathlib import Path
from doc_parse import call_document_parse_langchain
from lang import process_student_data
from utils import Utils

def initialize_session_state():
    """세션 상태 초기화"""
    if 'input_path' not in st.session_state:
        st.session_state.input_path = str(Path(__file__).parent / 'data' / 'raw' / 'pdf')
    if 'output_path' not in st.session_state:
        st.session_state.output_path = str(Path(__file__).parent / 'save')

def main():
    initialize_session_state()

    # 제목과 설명
    st.title("PDF 파싱 및 요약 시스템")
    st.markdown("""
    이 시스템은 PDF 문서를 파싱하고 요약하는 두 단계로 구성되어 있습니다:
    1. PDF 파싱 및 OCR
    2. 파싱된 파일 처리 및 요약
    """)

    # 기본 경로 설정
    base_path = Path(__file__).parent
    default_paths = {
        'html': str(base_path / 'save' / 'html'),
        'db': str(base_path / 'data' / 'db')
    }

    # 사이드바에 경로 설정
    st.sidebar.header("📁 경로 설정")
    
    # 입력 폴더 선정
    st.sidebar.markdown("### 입력 폴더 설정")
    input_path = st.sidebar.text_input(
        "입력 폴더 경로를 입력하세요:",
        value=st.session_state.input_path,
        help="PDF 파일이 있는 폴더의 경로를 입력하세요"
    )
    if input_path != st.session_state.input_path:
        st.session_state.input_path = input_path

    # 출력 폴더 설정
    st.sidebar.markdown("### 출력 폴더 설정")
    output_path = st.sidebar.text_input(
        "출력 폴더 경로를 입력하세요:",
        value=st.session_state.output_path,
        help="결과 파일이 저장될 폴더의 경로를 입력하세요"
    )
    if output_path != st.session_state.output_path:
        st.session_state.output_path = output_path

    # 선택된 폴더 정보 표시
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 선택된 폴더 정보")
    if os.path.exists(input_path):
        pdf_files = list(Path(input_path).glob("*.pdf"))
        st.sidebar.text(f"입력 폴더 PDF 파일 수: {len(pdf_files)}")
        if pdf_files:
            st.sidebar.markdown("#### PDF 파일 목록:")
            for pdf in pdf_files[:5]:  # 처음 5개만 표시
                st.sidebar.text(f"- {pdf.name}")
            if len(pdf_files) > 5:
                st.sidebar.text("...")
    else:
        st.sidebar.warning("입력 폴더가 존재하지 않습니다")

    if os.path.exists(output_path):
        json_files = list(Path(output_path).glob("*.json"))
        st.sidebar.text(f"출력 폴더 JSON 파일 수: {len(json_files)}")
    else:
        st.sidebar.warning("출력 폴더가 존재하지 않습니다")

    # 메인 영역 구성
    st.header("1️⃣ PDF 파싱 및 OCR")
    st.markdown("PDF 파일을 파싱하고 OCR 처리를 수행합니다.")
    
    # PDF 파싱 섹션
    if st.button("PDF 파싱 시작", key="parse_pdf"):
        if not os.path.exists(input_path):
            st.error(f"입력 경로가 존재하지 않습니다: {input_path}")
        else:
            pdf_files = list(Path(input_path).glob("*.pdf"))
            if not pdf_files:
                st.warning("처리할 PDF 파일이 없습니다.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, pdf_file in enumerate(pdf_files):
                    status_text.text(f"처리 중: {pdf_file.name}")
                    try:
                        call_document_parse_langchain(str(pdf_file), default_paths['html'])
                        progress_bar.progress((i + 1) / len(pdf_files))
                    except Exception as e:
                        st.error(f"파일 처리 중 오류 발생: {pdf_file.name}\n{str(e)}")
                
                st.success("PDF 파싱이 완료되었습니다!")

    # 구분선
    st.markdown("---")

    # 파일 처리 및 요약 섹션
    st.header("2️⃣ 파일 처리 및 요약")
    st.markdown("파싱된 파일을 처리하고 요약을 생성합니다.")
    
    if st.button("파일 처리 시작", key="process_files"):
        html_files = Utils.list_files(default_paths['html'], 'html')
        if not html_files:
            st.warning("처리할 HTML 파일이 없습니다.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, html_file in enumerate(html_files):
                status_text.text(f"처리 중: {Path(html_file).name}")
                try:
                    process_student_data(
                        html_file, 
                        Path(st.session_state.input_path),
                        output_dir=Path(st.session_state.output_path),
                        db_path=Path(default_paths['db'])
                    )
                    progress_bar.progress((i + 1) / len(html_files))
                except Exception as e:
                    st.error(f"파일 처리 중 오류 발생: {Path(html_file).name}\n{str(e)}")
            
            st.success("파일 처리가 완료되었습니다!")

    # 처리 결과 표시
    st.markdown("---")
    st.header("📊 처리 결과")
    
    # 결과 파일 목록 표시
    if st.button("결과 확인"):
        if os.path.exists(st.session_state.output_path):
            files = list(Path(st.session_state.output_path).glob("*.json"))
            if files:
                st.write("생성된 파일 목록:")
                for file in files:
                    st.text(file.name)
            else:
                st.info("생성된 파일이 없습니다.")
        else:
            st.warning("출력 폴더가 존재하지 않습니다.")

if __name__ == "__main__":
    main() 