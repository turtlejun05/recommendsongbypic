import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os # 파일 경로 존재 여부를 확인하기 위해 os 모듈을 추가..

# 감정 영어 매핑 및 템플릿
emotion_to_english = {
    "슬픈": "sad",
    "설레는": "excited",
    "평화로운": "peaceful",
    "신나는": "cheerful",
    "무거운": "intense"
}

templates = [
    "A photo that looks {emotion}.",
    "An image evoking a {emotion} feeling.",
    "A person appearing {emotion} in the scene."
]

# 키워드 리스트와 우선순위 정의
keywords = [
    # 감정
    "행복한", "웃는 얼굴", "기쁜 표정", "슬픈", "눈물 흘리는", "외로운", "우울한", "평화로운", "설레는", "감동적인",
    "불안한", "분노한", "놀란 얼굴", "사랑스러운", "따뜻한 마음", "지친 얼굴", "무표정한", "활기찬", "만족스러운",
    "신나는", "감성적인", "무거운", 
    
    # 배경/장소
    "푸른 바다", "해변", "숲속 길", "자연 풍경", "도시 거리", "하늘과 구름", "사막", "눈 덮인 산", "강가", "호수",
    "고요한 마을", "번화한 거리", "벚꽃길", "해넘이 풍경", "등산로", "밤의 도심", "별이 빛나는 하늘", "학교 운동장", "도서관", "지하철역", 
    
    # 날씨
    "맑은 하늘", "흐린 날", "비 오는 거리", "안개 낀 거리", "눈 내리는 풍경", "무지개가 뜬 하늘", "천둥 번개", "태풍이 몰아치는 하늘", "건조한 날씨", "바람 부는 들판",
    "따뜻한 햇살", "쌀쌀한 바람", "이슬 맺힌 풀", "햇빛이 반짝이는 거리", "구름 낀 하늘", "노을이 지는 하늘", "밤에 내리는 비", "서리 낀 창문", "잔잔한 바람", "습한 공기", 
    
    # 시간/계절
    "아침 햇살", "정오의 태양", "해질 무렵", "밤의 정적", "새벽 풍경", "여름 바다", "겨울 산", "봄의 꽃길", "가을 단풍", "해가 뜨는 순간",
    "불꽃놀이 밤", "크리스마스 분위기", "눈 쌓인 거리", "초여름 오후", "늦가을 공원", "비오는 오후", "밤하늘 별빛", "찬란한 오후 햇살", "저녁 노을", "졸린 새벽"
]


emotion_priority = ["슬픈", "설레는", "평화로운", "신나는", "무거운"]

# 한국어 기타 키워드 영어 매핑
other_keyword_map = {
    "푸른 바다": "blue sea", "해변": "beach", "숲속 길": "forest trail", "자연 풍경": "natural landscape",
    "도시 거리": "urban street", "하늘과 구름": "sky and clouds", "사막": "desert", "눈 덮인 산": "snowy mountain",
    "강가": "riverside", "호수": "lake", "고요한 마을": "quiet village", "번화한 거리": "busy street",
    "벚꽃길": "cherry blossom road", "해넘이 풍경": "sunset scene", "등산로": "hiking trail", "밤의 도심": "night city",
    "별이 빛나는 하늘": "starry sky", "학교 운동장": "school field", "도서관": "library", "지하철역": "subway station",
    "맑은 하늘": "clear sky", "흐린 날": "cloudy day", "비 오는 거리": "rainy street", "안개 낀 거리": "foggy street",
    "눈 내리는 풍경": "snowy scene", "무지개가 뜬 하늘": "rainbow in the sky", "천둥 번개": "thunderstorm",
    "태풍이 몰아치는 하늘": "stormy sky", "건조한 날씨": "dry weather", "바람 부는 들판": "windy field",
    "따뜻한 햇살": "warm sunlight", "쌀쌀한 바람": "chilly wind", "이슬 맺힌 풀": "dewy grass",
    "햇빛이 반짝이는 거리": "sunlit street", "구름 낀 하늘": "cloudy sky", "노을이 지는 하늘": "sunset sky",
    "밤에 내리는 비": "rain at night", "서리 낀 창문": "frosted window", "잔잔한 바람": "gentle breeze",
    "습한 공기": "humid air", "아침 햇살": "morning sunlight", "정오의 태양": "noon sun", "해질 무렵": "dusk",
    "밤의 정적": "quiet night", "새벽 풍경": "dawn scene", "여름 바다": "summer sea", "겨울 산": "winter mountain",
    "봄의 꽃길": "spring flower path", "가을 단풍": "autumn leaves", "해가 뜨는 순간": "sunrise",
    "불꽃놀이 밤": "fireworks night", "크리스마스 분위기": "Christmas vibe", "눈 쌓인 거리": "snow-covered street",
    "초여름 오후": "early summer afternoon", "늦가을 공원": "late autumn park", "비오는 오후": "rainy afternoon",
    "밤하늘 별빛": "night sky with stars", "찬란한 오후 햇살": "radiant afternoon sunlight", "저녁 노을": "evening glow",
    "졸린 새벽": "sleepy dawn"
}

# CLIP 모델 로딩 (처음 한번만)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_keywords_from_image(image: Image.Image):
    """이미지 받아서 분위기 키워드 3개 추출 (성능 강화: 영어 프롬프트 사용)"""
    # 감정 프롬프트 구성
    prompt_texts = []
    for kor in emotion_priority:
        eng = emotion_to_english.get(kor)
        if not eng:
            continue
        for tpl in templates:
            prompt_texts.append(tpl.format(emotion=eng))

    inputs = processor(text=prompt_texts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    # 감정별 평균 점수 계산
    scores = []
    for kor in emotion_priority:
        eng = emotion_to_english.get(kor)
        idxs = [i for i, t in enumerate(prompt_texts) if eng in t]
        if not idxs:
            scores.append(-1)
            continue
        vals = probs[idxs]
        scores.append(vals.mean().item())

    top_i = int(torch.tensor(scores).argmax())
    top_emotion = emotion_priority[top_i]

    # 나머지 키워드는 기존 한국어 리스트 중 감정 외 항목에서 추출 (영어 매핑 사용)
    remaining = [k for k in keywords if k not in emotion_priority]
    english_remaining = [other_keyword_map.get(k, k) for k in remaining]

    inputs2 = processor(text=english_remaining, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits2 = model(**inputs2).logits_per_image.softmax(dim=1)[0]
    top2 = torch.topk(logits2, k=2).indices.tolist()
    top_others = [remaining[i] for i in top2]

    return [top_emotion] + top_others

def get_top_emotion(image_path):
    """이미지에서 키워드 3개 추출 (감정 1개 + 기타 2개)"""
    try:
        # 파일을 열기 전에 해당 경로에 파일이 실제로 존재하는지 확인
        if not os.path.exists(image_path):
            print(f"경고: 지정된 이미지 경로에 파일이 없어요: {image_path}")
            # 파일이 없으면 에러 메시지를 포함한 리스트를 반환해서 서버에 알려주기
            return ["파일 없음", "경로 확인", "재시도"] 
        
        image = Image.open(image_path) # <<< 이 부분에서 에러가 날 수 있다그래서 try로 감쌈
        result = get_keywords_from_image(image)
        return result
    except OSError as e:
        # [Errno 22] Invalid argument와 같은 OSError가 발생하면 여기로
        print(f"에러: 이미지 '{image_path}'를 열 수 없어요. 원인: {e}")
        # 서버에서 이 에러를 처리할 수 있도록 에러 메시지를 담은 리스트를 반환
        return ["이미지 파일 오류", "경로 또는 이름 문제", "확인 필요"]
    except Exception as e:
        # 혹시 모르니까.. 오류가 너무 많이 나서
        print(f"에러: 이미지 '{image_path}' 처리 중 예상치 못한 오류 발생: {e}")
        return ["알 수 없는 오류", "관리자 문의", "재시도"]

__all__ = ["get_top_emotion"]

# 아래 테스트 코드는 완전히 삭제하거나, 주석 처리하세요.
# if __name__ == "__main__":
#     print(get_top_emotion("sample.jpg"))
