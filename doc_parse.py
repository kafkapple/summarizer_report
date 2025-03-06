"""
Requirements: `pip install pymupdf` to import fitz
"""
 
import os
import fitz
import pickle
import json
from dotenv import load_dotenv
import sys
from bs4 import BeautifulSoup
from tqdm import tqdm

from langchain_upstage import UpstageDocumentParseLoader
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

#from summarizer.utils import Utils
from utils import Utils
from pathlib import Path
import pickle

src_path = os.path.join(current_dir, 'src')
load_dotenv(os.path.join(src_path, '.env'))
os.environ["UPSTAGE_API_KEY"] = os.getenv("UPSTAGE_API")

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BaseMetadata:
    """문서 메타데이터의 기본 클래스"""
    def __init__(self, **kwargs):
        self._metadata = kwargs
        
    def get(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        self._metadata[key] = value
        
    def to_dict(self) -> Dict:
        return self._metadata.copy()

class DocumentParser(ABC):
    """일반화된 문서 파서 기본 클래스"""
    
    @abstractmethod
    def parse_metadata(self, content: str) -> BaseMetadata:
        """문서의 메타데이터를 파싱"""
        pass
    
    @abstractmethod
    def parse_content(self, content: str) -> Dict[str, Any]:
        """문서의 주요 내용을 파싱"""
        pass
    
    @abstractmethod
    def render(self, data: Dict[str, Any], output_file: str) -> None:
        """파싱된 데이터를 렌더링"""
        pass

class StudentMetadata(BaseMetadata):
    """학생 문서 메타데이터"""
    @property
    def student_id(self) -> str:
        return self.get('student_id', '')
        
    @property
    def name(self) -> str:
        return self.get('name', '')

class StudentRecordParser(DocumentParser):
    """학생 생활기록부 전용 파서"""
    
    def parse_metadata(self, content: str) -> tuple[str, str]:
        """문서의 메타데이터를 파싱"""
        soup = BeautifulSoup(content, 'html.parser')
        
        # BeautifulSoup 객체를 문자열로 변환
        #text_content = soup.get_text()
        
        # 문자열로 변환된 내용으로 파싱

        if td := soup.find('td', string='학번 성명'):
            student_id, name = extract_and_format_id_name(soup)
        else:
            try:
                student_id = self._extract_student_id(soup)
                name = self._extract_name(soup)
            except:
                student_id = ""
                name = ""
        
        return student_id, name
    
    def _extract_student_id(self, soup: BeautifulSoup) -> str:
        """학번 추출 및 정제 로직"""
        td = soup.find('td', string='학번')
        raw_id = td.find_next('td').text.strip()
        return self._clean_student_id(raw_id)
            
    def _extract_name(self, soup: BeautifulSoup) -> str:
        """학생 이름 추출 및 정제 로직"""
        td = soup.find('td', string='성명')

        name = td.find_next('td').text.strip()
        return name


    def _clean_student_id(self, raw_id: str) -> str:
        """학번 정제 로직"""
        # 숫자만 추출
        return ''.join(filter(str.isdigit, raw_id))
    
    def parse_content(self, content: str) -> Dict[str, Any]:
        """문서의 주요 내용을 파싱"""
        # BeautifulSoup을 사용해서 문서 내용을 파싱하는 로직 구현
        soup = BeautifulSoup(content, 'html.parser')
        # 필요한 데이터 추출 및 처리
        parsed_data = {}  # 실��� 구현에서는 필요한 데이터를 담아 함
        return parsed_data
    
    def render(self, data: Dict[str, Any], output_file: str) -> None:
        """파싱된 데이터를 렌더링"""
        # 파싱된 데이터를 원하는 형식으로 출력하는 로직 구현
        # 예: HTML, JSON 등으로 저장
        pass



def split_pdf(input_file, save_path, batch_size):
    # Open input_pdf
    input_pdf = fitz.open(input_file)
    num_pages = len(input_pdf)
    print(f"Total number of pages: {num_pages}")
 
    # Split input_pdf
    for start_page in range(0, num_pages, batch_size):
        end_page = min(start_page + batch_size, num_pages) - 1
 
        # Write output_pdf to file
        input_file_basename = os.path.splitext(input_file)[0]
        output_file = os.path.join(save_path, f"{input_file_basename}_{start_page}_{end_page}.pdf")
        print(output_file)
        with fitz.open() as output_pdf:
            output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
            output_pdf.save(output_file)
 
    # Close input_pdf
    input_pdf.close()
 

 #UpstageDocumentParseLoader는 file_path 매개변수를 추가로 받아 로드할 파일 경로를 지정합니다24.
#UpstageDocumentParseParser는 파일 경로 대신 Blob 객체를 ���하는 데 특화
def call_document_parse_langchain(input_file, save_path=None):
    """
    문서를 파싱하고 결과를 저장하는 함수
    
    Args:
        input_file: 입력 파일 경로
        save_path: 저장할 경로 (기본값: None)
    """
    loader = UpstageDocumentParseLoader(
        file_path=input_file,
        split="page",
        ocr="force",
        output_format="html",
        coordinates=True,
        base64_encoding=["paragraph", "table", "equation"]
    )
    pages = loader.load()
    
    print(f'Finished...')
    
    # save_path가 제공되지 않은 경우 기본 경로 사용
    if save_path is None:

        save_path = Path(current_dir) / Path(input_file).stem + '.pkl'
    else:
        save_path = Path(save_path) / (Path(input_file).stem + '.pkl')
    
    # 저장 디렉토리가 없으면 생성
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    pickle.dump(pages, open(save_path, 'wb'))
    print(f'Save...{save_path}')

def json_to_html(save_path='save'):
    json_files = Utils.list_files(save_path, extension='json')
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as file:
            json_data = json.load(file)
        # JSON 데이터를 HTML 형태로 변환 (필요시 키 값 추출 및 문자열 조합)
        try:
            html_content = json_data['content'].get("html", "")
            if isinstance(html_content, dict):
                html_content = json.dumps(html_content)  # 딕셔너리를 문자열로 변환
        except:
            html_content = json.dumps(json_data)  # 전체 JSON 데이터를 문자열로 변환
        # HTML 파일로 저장
        with open(os.path.join(save_path, os.path.splitext(json_file)[0]+'.html'), "w", encoding="utf-8") as html_file:
            html_file.write(html_content)

def load_json_and_save(data_path='data', save_path='save'):
    files = Utils.list_files(data_path, extension='json')
    parser = StudentRecordParser()  # StudentRecordParser 인스턴스 생성
    for file in files:
        print(file)
        output_file = os.path.join(save_path, os.path.splitext(os.path.basename(file))[0])
        parser.pipeline_document(file, target_formats=['html', 'markdown'], save_path=output_file)

from pathlib import Path

def parse_and_save_html(pkl, save_path):
    print(f'pkl: {pkl}')
    print(f'save_path: {save_path}')
    save_path = Path(save_path) / Path(pkl).stem
    os.makedirs(save_path, exist_ok=True)
    loaded_data = pickle.load(open(pkl, 'rb'))
    
    buffer = []
    current_content = ""
    current_student = None
    parser = StudentRecordParser()
    file_counter = 1  # 학번/이름을 찾지 못했을 경우를 위한 카운터
    
    SECTION_MARKER = "<수학 탐구활동 및 보고서 작성>"
    
    def save_current_document():
        nonlocal current_student, current_content, file_counter
        if current_content:
            # 현재 내용에서 학번과 이름 찾기
            student_id, name = parser.parse_metadata(current_content)
            
            # 학번과 이름을 찾지 못한 경우
            if not student_id or not name:
                student_id = f"unknown_{file_counter:03d}"
                name = "unknown"
                file_counter += 1
            
            # 파일 저장
            document = {'student_id': student_id, 'name': name, 'contents': current_content}
            save_to_html(document, Path(save_path) / f"{student_id}_{name}.html")
            current_content = ""
    
    for page in tqdm(loaded_data):
        page_content = page.page_content
        
        # 새로운 섹션 마커를 찾으면
        if SECTION_MARKER in page_content:
            # 이전 문서 저장
            save_current_document()
            # 새로운 문서 시작
            current_content = page_content
        else:
            # 현재 문서에 페이지 내용 추가
            current_content += page_content
    
    # 마지막 문서 저장
    save_current_document()


# # Load the JSON data
# file_path = '/mnt/data/.json'
# with open(file_path, 'r', encoding='utf-8') as file:
#     student_data = json.load(file)

# Function to render HTML
def render_html_with_latex_from_json(data, student_id, name, output_file="output"):
    analysis = data.get("analysis", {})
    output_file = os.path.join(output_file, f"{student_id}_{name}.html")
    
    html_content = f"""
    <h2>학생 생활기록부 요약</h2>
    <table>
        <tr><th>학번</th><td>{student_id}</td></tr>
        <tr><th>이름</th><td>{name}</td></tr>
    </table>
    <h3>핵심 키워드</h3>
    <ul>
        {''.join(f'<li>{kw["keyword"]} (중요도: {kw["relevance"]})</li>' for kw in analysis.get("keywords", []))}
    </ul>
    <h3>학문적 락</h3>
    <ul>
        {''.join(f'<li>{context["topic"]}: {context["connection"]} (중요도: {context["importance"]})</li>' for context in analysis.get("academic_context", []))}
    </ul>
    <h3>사고 과정</h3>
    <p><strong>동기:</strong> {analysis.get("thought_process", {}).get("motivation", "N/A")}</p>
    <p><strong>접근:</strong> {analysis.get("thought_process", {}).get("approach", "N/A")}</p>
    <p><strong>수행:</strong> {analysis.get("thought_process", {}).get("execution", "N/A")}</p>
    <p><strong>결과:</strong> {analysis.get("thought_process", {}).get("result", "N/A")}</p>
    <h3>최종 보고서</h3>
    <h4>{analysis.get("final_report", {}).get("title", "N/A")}</h4>
    <p>{analysis.get("final_report", {}).get("content", "N/A")}</p>
    <h4>평가 항목</h4>
    <ul>
        {''.join(f'<li>{point["category"]}: {point["description"]}</li>' for point in analysis.get("final_report", {}).get("evaluation_points", []))}
    </ul>
    """
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>학생 생활기록부 요약</title>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            ul {{
                list-style-type: none;
                padding: 0;
            }}
            ul li {{
                margin: 5px 0;
            }}
        </style>
    </head>
    <body>
        {html_content}
        <script type="text/javascript">
            if (window.MathJax) {{
                window.MathJax.typesetPromise();
            }}
        </script>
    </body>
    </html>
    """
    
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(html_template)
    return f"HTML 파일이 되었습니다: {output_file}"

class HTMLRenderer:
    """JSON 데이터를 HTML로 렌더링하는 클래스"""
    
    @staticmethod
    def _convert_to_latex(text):
        """텍스트를 LaTeX 형식으로 변환"""
        return text.replace('$', '\\(').replace('$', '\\)')
    
    @staticmethod
    def _generate_style():
        """HTML 스타일 정의"""
        return """
        <style>
            body { font-family: Arial, sans-serif; margin: 2em; }
            .container { max-width: 800px; margin: auto; }
            .section { margin-bottom: 2em; }
            .section-title { 
                font-size: 1.2em; 
                font-weight: bold;
                margin-bottom: 1em;
                border-bottom: 2px solid #333;
                padding-bottom: 0.5em;
            }
            .item { margin-bottom: 1em; }
            .item-title { font-weight: bold; }
            .array-item { margin-left: 1em; margin-bottom: 0.5em; }
            .nested-object { margin-left: 1em; }
        </style>
        """

    def _render_value(self, value, level=0):
        """값을 HTML로 렌더링"""
        indent = "    " * level
        if isinstance(value, list):
            items = [f'<div class="array-item">{self._convert_to_latex(str(item))}</div>'
                    for item in value]
            return f'\n{indent}'.join(items)
        elif isinstance(value, dict):
            return self._render_object(value, level + 1)
        else:
            return self._convert_to_latex(str(value))

    def _render_object(self, obj, level=0):
        """객체를 HTML로 렌더링"""
        html_parts = []
        indent = "    " * level
        
        for key, value in obj.items():
            if key.startswith('_'):  # 언더스코어로 시작하는 필드는 건너뜀
                continue
                
            formatted_key = key.replace('_', ' ').title()
            html_parts.append(f'{indent}<div class="item">')
            html_parts.append(f'{indent}    <div class="item-title">{formatted_key}:</div>')
            html_parts.append(f'{indent}    <div class="item-content">')
            html_parts.append(f'{indent}        {self._render_value(value, level + 2)}')
            html_parts.append(f'{indent}    </div>')
            html_parts.append(f'{indent}</div>')
        
        return '\n'.join(html_parts)

    def render(self, data, title=""):
        """JSON 데이터를 완전한 HTML 문서로 렌더링"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    {self._generate_style()}
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        {self._render_object(data)}
    </div>
</body>
</html>
"""
        return html

def render_html_from_json(json_data, student_id="", name="", output_dir="output"):
    """JSON 데이터로부터 HTML 보고서 생성"""
    renderer = HTMLRenderer()
    title = f"학생 보고서 - {student_id} {name}" if student_id and name else "학생 보고서"
    html_content = renderer.render(json_data, title=title)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{student_id}_{name}_report.html")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML 보고서 생성 완료: {output_file}")

def extract_and_format_id_name(soup: BeautifulSoup) -> tuple[str, str]:
    """
    문자열에서 괄호와 숫자, 이름을 추출하고 포맷팅하는 함수
    
    Args:
        soup: BeautifulSoup 객체
    
    Returns:
        tuple[str, str]: (포맷팅된 학번, 추출된 이름)
    """
    import re
    
    # 다양한 패턴의 td 태그 찾기
    td = soup.find('td', string=re.compile(r'학[번변].*성명'))
    if not td:
        return "", ""
        
    text = td.find_next('td').text.strip()
    print(f'Test: {text}')
    
    # 모든 공백과 소숫점 제거
    text = ''.join(text.split()).replace('.', '')
    
    # 괄호 쌍과 숫자 찾기
    full_pattern = r'\((\d+)\)'  # 정상적인 괄호 패턴
    single_pattern = r'(?:\((\d+)|(\d+)\))'  # 한쪽 괄호만 있는 패턴
    
    # 정상적인 괄호 쌍 먼저 찾기
    numbers = re.findall(full_pattern, text)
    
    # 괄호 쌍이 충분하지 않은 경우, 한쪽 괄호 패턴도 검사
    if len(numbers) < 2:
        single_matches = re.finditer(single_pattern, text)
        for match in single_matches:
            num = match.group(1) or match.group(2)  # 둘 중 매치된 그룹 사용
            if num and num not in numbers:
                numbers.append(num)
    
    # 이름 추출 (마지막 괄호 안의 내용)
    name_pattern = r'\(([^)]+)\)'
    name_matches = re.findall(name_pattern, text)
    name = name_matches[-1] if name_matches else ""
    
    # 이름에서 '01'을 '이'로 변환하고 숫자와 특수문자 제거
    name = name.replace('01', '이')
    name = re.sub(r'[^가-힣]', '', name)  # 한글만 남기기
    
    if len(numbers) >= 2:
        try:
            # 번호 처리 (마지막 숫자, 최대 2자리)
            last_num = str(int(numbers[-1]))
            if len(last_num) > 2:
                last_num = last_num[-2:]  # 3자리 이상이면 마지막 2자리만 사용
            
            # 반 번호 처리 (끝에서 두 번째 숫자, 1자리)
            class_num = str(int(numbers[-2]))
            if len(class_num) > 1:
                if class_num.startswith('1'):  # 첫자리가 1이면 제외
                    class_num = class_num[1:]
                else:
                    class_num = class_num[-1]  # 마지막 자리만 사용
            
            # 학번 생성 (1학년 + 반 + 번호)
            student_id = f"1{class_num.zfill(2)}{last_num.zfill(2)}"
            
            # 최종 검증: student_id가 숫자로만 ���성되어 있는지 확인
            if not student_id.isdigit():
                return "", ""
                
            return student_id, name
            
        except (ValueError, AttributeError, IndexError):
            return "", ""
    
    return "", ""

def save_to_html(html_content: str, file_path: str) -> None:
    """
    HTML 내용을 파일로 저장
    
    Args:
        html_content: 저장할 HTML 문자열
        file_path: 저장할 파일 경로
    """
    try:
        # HTML 기본 구조 추가
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Generated HTML</title>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # 파일로 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
            
    except Exception as e:
        print(f"Error saving HTML file: {e}")

def main():
    file_name = Path('.pdf')
    data_pdf_path = Path(__file__).parent / 'data'
    data_path = Path(__file__).parent / 'data' / file_name #'processed' / '1_pages'
    save_path = Path(__file__).parent / 'data' / 'processed' #/ 'html'
    #save_html_path = Path(__file__).parent / 'save' / 'html'

    # html_sample = Path(__file__).parent / 'save' / 'html' / ".html"
    # json_file_path = Path(__file__).parent / 'save' / '.json'
    
    # ### 1JSON 파일을 읽어오기
    # with open(json_file_path, "r", encoding="utf-8") as file:
    #     json_data = json.load(file)
    # render_html_with_latex_from_json(json_data, , save_path / ".html")

# 1. LangChain Document Parse API

   # os.makedirs(data_path, exist_ok=True)
    call_document_parse_langchain(input_file=data_path, save_path=save_path) # 1. Document Parse API
    
    # data_pdfs = Utils.list_files(data_pdf_path, extension='pdf')
    # for pdf in data_pdfs:
    # #    call_document_parse_langchain(input_file=pdf)
    #     print(pdf)
    # data_processed_path = Path(__file__).parent / 'data' / 'processed'
    # data_pkl = Utils.list_files(data_processed_path, extension='pkl')
    # for pkl in data_pkl:
    #     print(pkl)
    
    #2. Parse more
        
    parse_and_save_html(save_path /f"{file_name.stem}.pkl", save_path / "html")

    # data_json_path = os.path.join(current_dir, 'data_json')
    #load_json_and_save(data_path=data_json_path, save_path=save_path)

    
if __name__ == "__main__":
    main()

