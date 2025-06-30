from flask import Flask, render_template, request
from PIL import Image
import torch
import clip
from loadfile import recommend_song

app = Flask(__name__)

# CLIP 모델 로딩
device = "cuda" if torch.cuda.is_available() else "cpu" # Gpu가 있으면 cuda, 없으면 cpu로 설정
model, preprocess = clip.load("ViT-B/32", device=device)

@app.route("/", methods=["GET", "POST"])
def home():
    # 초기 변수들
    # 처음엔 결과 섹션 안보이도록
    
    mood = None
    recs = None
    error_message = None
    
    if request.method == "POST":
        try:
            # 이미지 받기
            if 'image' not in request.files:
                error_message = "이미지 파일을 선택해주세요."
            else:
                file = request.files["image"]
                if file.filename == '':
                    error_message = "이미지 파일을 선택해주세요."
                else:
                    # 이미지 처리
                    image = Image.open(file.stream).convert("RGB")
                    
                    # CLIP 전처리 → 벡터 추출
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_features = model.encode_image(image_input)
                    
                    # 추천 시스템 호출
                    from loadfile import mood_classifiyer
                    values = mood_classifiyer[image_features]
                    danceability, energy, loudness, acousticness, instrumentalness, valence, tempo = values
                    
                    # 추천 함수 호출
                    mood, recs = recommend_song(danceability, energy, loudness, acousticness, 
                                              instrumentalness, valence, tempo)
                    
        except Exception as e:
            error_message = f"처리 중 오류가 발생했습니다: {str(e)}"
    
    # 결과를 home.html에 전달
    return render_template("home.html", 
                         mood=mood, 
                         recommendations=recs, 
                         error=error_message)

if __name__ == "__main__":
    app.run(debug=True)