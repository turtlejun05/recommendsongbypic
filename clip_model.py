from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# 키워드 리스트와 우선순위 정의 (기존과 동일)
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

# CLIP 모델 로딩 (처음 한번만)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_keywords_from_image(image: Image.Image):
    """이미지 받아서 분위기 키워드 3개 추출"""
    inputs = processor(text=keywords, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
    
    # 감정 키워드 중 가장 높은 것
    emotion_indices = [keywords.index(k) for k in emotion_priority]
    emotion_scores = probs[emotion_indices]
    top_emotion_idx = emotion_indices[torch.argmax(emotion_scores)]
    top_emotion = keywords[top_emotion_idx]
    
    # 나머지 키워드 중 상위 2개
    probs[top_emotion_idx] = -1
    top_other_indices = torch.topk(probs, k=2).indices.tolist()
    top_others = [keywords[i] for i in top_other_indices]
    
    return [top_emotion] + top_others

def get_top_emotion(image_path):
    """이미지에서 키워드 3개 추출 (감정 1개 + 기타 2개)"""
    image = Image.open(image_path)
    result = get_keywords_from_image(image)
    return result  # 키워드 3개 리스트 반환

if __name__ == "__main__":
    # 테스트
    print(get_top_emotion("sample.jpg"))