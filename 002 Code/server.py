from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any
import json
from PIL import Image, ImageDraw, ImageFont
import base64
import io
from paddleocr import PaddleOCR, draw_ocr
import numpy as np

# 환경변수 로드 및 GPT 클라이언트 초기화
load_dotenv(dotenv_path=".env")
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
app = FastAPI()
ocr = PaddleOCR(use_angle_cls=True, lang='korean')

# 디버깅용 출력
print("🔑 API KEY 불러오기 상태:", "성공" if api_key else "실패")
print("🔑 API KEY 일부:", api_key[:10] if api_key else "None")

if not api_key:
    raise ValueError("❌ 환경변수 OPENAI_API_KEY를 찾을 수 없습니다.")

# 요청 형식 정의
class AnalyzeRequest(BaseModel):
    user_prompt: str
    image_base64: str  # base64로 인코딩된 이미지 문자열

# PaddleOCR로 이미지에서 텍스트 박스 추출
def extract_ocr_boxes(image_base64: str) -> List[Dict[str, Any]]:
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    result = ocr.ocr(image_np, cls=True)

    extracted = []
    for line in result:
        for box in line:
            text = box[1][0]
            coords = box[0]  # [[x1, y1], ..., [x4, y4]]
            extracted.append({"text": text, "box": coords})
    return extracted

# GPT 호출 함수
def call_gpt(user_prompt, ocr_boxes, base64_img):
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": 
"""
너는 user_prompt로 들어오는 사용자 예약 요청을 보고, 필요한 부분에 박스를 그려주는 AI야. 
4개의 점을 다 찍어서 꼭 박스로 그려줘. 삼각형이나 선으로 그리면 안 돼.
박스의 y축들의 값은 두 쌍씩 같아야 해. 그 쌍은 x축만 달라.
각 끝부분의 4개의 점으로 박스로 표현해.

만약 지역명이 있다면 지역명에 박스를 쳐줘야해
"출발지, 도착지"라는 글씨는 박스를 그리면 안 돼
출발지와 도착지에 써있는 지역이 user_prompt의 출발, 도착 부분이 다르다면 박스를 그려줘
만약 출발일, 관람일 등 날짜와 시간이 있다면 하나의 박스로 합쳐서 그려줘
인원이 있는 경우는 총 몇 명이라고 써져있는 부분에 박스를 그려줘
예매하기 버튼을 마지막으로 그려주면 돼
몇 명인지 숫자가 아니라 한글이여도 인식해서 박스 그려줘.


출력은 JSON 형식으로, 선택된 요소들의 text와 좌표 box를 포함해:

"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
다음은 사용자의 예약 요청이야: "{user_prompt}"

아래는 OCR을 통해 추출된 UI 요소와 해당 박스 좌표야:
{json.dumps(ocr_boxes, ensure_ascii=False)}

이미지를 보고, 어떤 요소를 눌러야 할지 판단해서 'selected_boxes' 필드에 해당 OCR 텍스트와 좌표를 JSON 형태로 반환해줘.
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content

@app.post("/analyze")
async def analyze_ui(request: AnalyzeRequest):
    try:
        # 항상 내부에서 OCR 실행
        ocr_boxes = extract_ocr_boxes(request.image_base64)
        result = call_gpt(request.user_prompt, ocr_boxes, request.image_base64)
        return json.loads(result)

    except Exception as e:
        print("❌ 에러:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
