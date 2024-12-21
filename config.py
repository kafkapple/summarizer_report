class JsonFunction:
    def __init__(self):
        self.json_function_final =[
                        {
            "name": "write_student_report",
            "description": "학생의 활동 내역 정보를 분석하여 생활기록부 특기사항을 작성.",
            "parameters": {
                "type": "object",
                "properties": {
                "activity_information": {
                        "type": "string",
                        "description": "작성된 생활기록부 특기사항 내용" #"학생의 활동 내역 정보를 바탕으로 평가 보고서 작성. 수행 내용 요약, 학습 동기와 활동 태도, 수행 과정의 관심과 태도, 학습 의지와 약점을 극복하려는 노력, 학습 활동의 성취 수준, 학생의 핵심 역량 등을 포함."
                }
                },
                "required": ["activity_information"]
            }
            }
        ]
        self.role = """
        당신은 수학 교과 전문가입니다. 학생의 활동 내역을 분석하여 생활기록부 특기사항을 작성하고, 결과를 OpenAI 함수 호출의 'activity_information' 필드에 포함시켜 반환하세요.
        """
        self.prompt = """
        다음 예시를 참고하여, 다음 조건을 만족하면서 학생의 수학 활동을 분석하고 생활기록부 특기사항을 작성:
        - 모든 내용을 개조식으로 작성.
        - '-' 와 같은 시작 기호는 포함하지 않고, 문장 끝을 '~함'과 같은 형태로 종결. 
        - 학생의 수행 과정과 성과를 구체적인 전문 용어를 사용하여 명확하고 객관적으로 기술. 
        - 이 활동 내역을 통해 학생의 역량이 드러나도록, 긍정적 관점에서 평가. 
        - 진로 희망과 논리적 연관성이 있는 활동의 경우 언급하되, 관련 내용이 없는 경우 언급하지 않음.
        - 모든 내용은 한국어로 작성. 
        - **결과는 함수 호출 형태로 출력하고, 작성된 내용을 'activity_information' 필드에 포함.**

        학생의 활동 내역: 
        """
        # self.example = """
        # 유리함수와 무리함수의 개념과 성질을 이용하여 조건에 맞는 그래프를 정확하게 그렸으며, 그래프를 통해 문제가 바로 해결된다는 점을 흥미롭게 생각하여 수업에 몰입하는 계기가 됨. 
        # 릴레이 풀이 모둠 활동에서 자신이 맡은 문제는 꼭 해결하여 팀원들에게 부담을 주지 않겠다는 마음가짐을 가짐. 모르는 부분을 모둠의 멘토역할인 솔브맨에게 주저하지 않고 질문하는 적극성을 보임.
        # 2학기에는 수학의 문제 해결력을 향상시키기위해 수학멘토링 프로그램에 의해 자발적으로참여하여 점심시간과 방과후 15분씩 주2회씩참여하는 노력을 함. 수업중 활동지를 완성하지 못하는 경우 늦더라고 끝까지 해내는 등 쉽게 포기하지않고 성장의지가 강함.
        # """

        self.example = """
        평소 읽던 기사에서 프랙탈 의학에 관하여 접하고 뇌의 주름, 뉴런 사이의 네트워크, 뇌파, 폐포 등에서 프랙탈의 구조를 발견할 수 있음에 이에 관한 탐구 활동을 진행함. 인체의 프랙탈 구조는 작은 공간에 단순한 법칙에 따라 많은 정보와 기능을 넣을 수 있는 효율적 구조라고 설명하고 폐포를 예를 들어 기체 교환의 효율을 높일 수 있는 이유를 논리적으로 설명함. 또한 프랙탈 차원이라는 용어를 접하고 이에 대한 심화 탐구를 진행함. '폐의 프랙탈 구조'를 주제로 선정하여 프랙탈의 수학적 근거를 분석하고 이를 통해 폐의 프랙탈 구조에 의한 효율성을 발표함. 특히 프랙탈 차원의 개념을 찾아 탐구하였으며 정육면체를 예시로 들어 받아드리기 쉽도록 설명하고 n차원 물체를 등분하는 상황으로 일반화하여 도형의 개수를 통해 차원을 정의함. 이러한 프랙탈 차원의 개념을 폐암 환자 판별에도 적용할 수 있음을 찾아내고 수학적 개념으로 실제 문제를 해결할 수 있는 사고력을 향상시킴.
        """
        # 명확하게 제시되어 있는 확률이 실제로 시행을 여러 번 하였을 때의 직관적으로 예측하는 값과 일치할지에 대한 궁금증을 해결하기 위하여 '모바일 게임에서 확률형 아이템의 고시 확률과 실제확률의 관한 연구'의 문헌을 읽고 도박사의 오류에 관하여 알게 된 바를 정리하여 발표함. 수학적 확률을 통계적 확률로 받아드리는 인간 심리에 의해 잘못된 판단을 할 수 있다고 설명하며 큰 수의 법칙을 따르기 위해서는 충분한 실험 횟수가 매우 중요함을 설명하고 이러한 문제 상황에는 독립시행의 상황으로 이항분포를 적용해야함을 설명하며 직관의 잘못된 부분을 분명하게 찾아내고 적합한 확률적 개념을 정확히 적용하는 모습을 보여줌. 또한 궁금증에서 그치지 않고 자율적인 문헌 탐색을 통해 스스로 해결하는 태도를 기름. 발표 이후 '도박사의 오류와 뜨거운 손 오류'에 대한 학급 친구들의 반응을 분석하여 실제로 얼마나 인지적 오류를 가지고 있는지 통계를 냄. 이를 위해 직접 문제 상황을 설정한 설문지를 만들어 조사하였으며 이항분포를 적용해야 한다는 것을 알고 있음에도 직관적으로 잘못된 오류를 범하는 것을 보여주며 사고의 오류 가능성을 인정하고 수학적 논리를 통해 이를 발전시킬 수 있다고 설명함.    
        
        self.struct = {'student_id': '', 'name': '', 'activity_info': ''}
        
import os
from dotenv import load_dotenv
import copy
import argparse
import logging
# httpx 로거의 레벨을 WARNING으로 설정하여 INFO 레벨 로그를 숨깁니다.
logging.getLogger("httpx").setLevel(logging.WARNING)
# 또는 전체 로깅 레벨을 설정할 수 있습니다.
#logging.basicConfig(level=logging.WARNING)


class Config:
    def __init__(self):
        # 기본 설정값
        self.INCLUDE_KEYWORDS = True
        self.INCLUDE_FULL_TEXT = False
        self.ENABLE_CHAPTERS = True
        
        # 환경 변수 및 기타 설정...
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.src_path = os.path.join(self.base_path, 'src')
        self.save_path = os.path.join(self.base_path, 'save')
        self.result_path = os.path.join(self.base_path, 'result')

        os.makedirs(self.src_path, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        
        load_dotenv(os.path.join(self.src_path, '.env'))
        
        self.YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.NOTION_TOKEN = os.getenv("NOTION_TOKEN")

        self.DIFFBOT_API_TOKEN = os.getenv("DIFFBOT_API_TOKEN")
        
        self.NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
        self.NOTION_DB_YOUTUBE_CH_ID = os.getenv("NOTION_DB_YOUTUBE_CH_ID")
        self.NOTION_DB_RAINDROP_ID = os.getenv("NOTION_DB_RAINDROP_ID")
        self.NOTION_DB_POCKET_ID = os.getenv("NOTION_DB_POCKET_ID")

        self.DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
        self.RAINDROP_TOKEN = os.getenv("RAINDROP_TOKEN")

        self.POCKET_CONSUMER_KEY = os.getenv("POCKET_CONSUMER_KEY")
        self.POCKET_ACCESS_TOKEN = os.getenv("POCKET_ACCESS_TOKEN")
        self.UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

        self.OUTPUT_LANGUAGE = 'ko'

        self.GPT_MODEL = 'gpt-4o' #'gpt-3.5-turbo'
        self.MAX_TOKEN = 4096
        self.max_token_response = 600
        self.min_token_response = 150
        self.TEMPERATURE = 0.2 # 0.1
        self.system_content = """You are a logical summary assistant. Follow these rules:
        1. Please respond in JSON format only. Do not include comments or any additional text.
        2. Cover different aspects without overlap.
        3. Use concise language.
        4. Maintain consistent formatting.
        """
   
        # 설정 로그 출력
        print("\n=== Configuration ===")
        print(f"Keywords Enabled: {self.INCLUDE_KEYWORDS}")
        print(f"Full Text Enabled: {self.INCLUDE_FULL_TEXT}")
        print(f"Chapters Enabled: {self.ENABLE_CHAPTERS}")

        # 스키마 생성 및 할당
        self.json_function_section, self.json_function_final, self.json_function_full = self.create_schema()
    
    def update_runtime_settings(self, keywords=None, full_text=None, chapters=None):
        """실행 시 설정 업데이트"""
        if keywords is not None:
            self.INCLUDE_KEYWORDS = keywords
        if full_text is not None:
            self.INCLUDE_FULL_TEXT = full_text
        if chapters is not None:
            self.ENABLE_CHAPTERS = chapters
            
        # 설정 로그 출력
        print("\n=== Configuration ===")
        print(f"Keywords Enabled: {self.INCLUDE_KEYWORDS}")
        print(f"Full Text Enabled: {self.INCLUDE_FULL_TEXT}")
        print(f"Chapters Enabled: {self.ENABLE_CHAPTERS}")
        print("==================\n")

# 1<= section <= 4
# 2<= bullet <= 5
# 50<= word <= 100
    # 공통 스키마 정의
    def create_schema(self):
        # Detailed descriptions for better control
        description_section = """Divide content into meaningful sections. Provide detailed section summaries following these rules:
        1. Each section should have 2-5 detailed bullet points
        2. Each bullet point should be 20-30 words long
        3. Focus on specific details and examples
        4. Avoid generic statements
        5. Include relevant numbers and facts when available"""

        description_full = """Create a concise overall summary following these rules:
        1. Maximum 3 bullet points for the entire text
        2. Each bullet point should be 15-20 words
        3. Focus on high-level key points only
        4. Avoid detailed examples
        5. Maintain broad perspective"""

        description_bullet = """Create detailed bullet points that:
        1. Are 20-30 words each
        2. Include specific examples or data
        3. Focus on distinct aspects"""

        # Schema for sections with length controls
        section_schema = {
            "sections": {
                "type": "array",
                "description": description_section,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Concise title (3-5 words)",
                            "maxLength": 30
                        },
                        "summary": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "minLength": 50,  # Enforce minimum length for detail
                                "maxLength": 100  # Enforce maximum length for clarity
                            },
                            "description": description_bullet,
                            "minItems": 1,
                            "maxItems": 4
                        }
                    },
                    "required": ["title", "summary"]
                },
                "minItems": 1
            }
        }

        # Schema for full summary with stricter length controls
        full_summary_schema = {
            "full_summary": {
                "type": "array",
                "items": {
                    "type": "string",
                    "minLength": 30,  # Shorter minimum for conciseness
                    "maxLength": 60   # Shorter maximum for brevity
                },
                "description": description_full,
                "minItems": 2,
                "maxItems": 3
            }
        }

        one_sentence_summary_schema = {
            "one_sentence_summary": {
                "type": "string",
                "description": "Single sentence capturing the main idea (15-20 words)",
                "minLength": 30,
                "maxLength": 50
            }
        }

        keyword_schema = {
            "keywords": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {
                            "type": "string",
                            "description": "Key concept or term",
                            "maxLength": 20
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of occurrences"
                        }
                    },
                    "required": ["term", "count"]
                },
                "maxItems": 3
            }
        }

        def create_function(properties_dict, required_fields):
            return [{
                "name": "create_summary",
                "description": "Generate structured summary with specified detail levels",
                "parameters": {
                    "type": "object",
                    "properties": properties_dict,
                    "required": required_fields
                }
            }]

        # Create function schemas with updated properties
        section_properties = {**section_schema}
        json_function_section = create_function(section_properties, ["sections"])

        final_properties = {**full_summary_schema, **one_sentence_summary_schema, **keyword_schema}
        json_function_final = create_function(final_properties, ["full_summary", "one_sentence_summary"])

        full_properties = {**section_schema, **full_summary_schema, **one_sentence_summary_schema, **keyword_schema}
        json_function_full = create_function(full_properties, ["sections", "full_summary", "one_sentence_summary"])

        return [json_function_section, json_function_final, json_function_full]
    