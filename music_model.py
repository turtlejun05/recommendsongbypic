import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# 데이터 로딩
df = pd.read_csv('data.csv')

# 특징 스케일링
features = df[['danceability','energy', 'loudness','acousticness','instrumentalness','valence', 'tempo']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 1. 입력 차원 정의
input_dim = X_scaled.shape[1]   # 예: 7 (음악 feature 개수)
encoding_dim = 3                # 축소할 차원 수

# 2. 입력 레이어
input_layer = Input(shape=(input_dim,))

# 3. 인코더 구조
encoded = Dense(encoding_dim, activation='relu', name="encoder")(input_layer)

# 4. 디코더 구조
decoded = Dense(input_dim, activation='linear', name="decoder")(encoded)

# 5. 전체 오토인코더 모델: 입력 → 인코딩 → 디코딩
autoencoder = Model(inputs=input_layer, outputs=decoded, name="autoencoder")

# 6. 인코더 단독 모델
encoder = Model(inputs=input_layer, outputs=encoded, name="encoder_model")

# 7. 디코더 단독 모델 구성
# 인코딩된 입력을 받아 복원하는 모델
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.get_layer("decoder")(encoded_input)
decoder = Model(inputs=encoded_input, outputs=decoder_layer, name="decoder_model")

# 8. 모델 컴파일 & 학습
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=512,
    shuffle=True,
    verbose=0
)

# 인코딩 & KMeans
X_encoded = encoder.predict(X_scaled)
kmeans = KMeans(n_clusters=5, random_state=42)
df['mood_cluster'] = kmeans.fit_predict(X_encoded)

#중심점 특징벡터
centroids = kmeans.cluster_centers_
centerfeatures = decoder.predict(centroids)

# print(centerfeatures)

# 감정 매핑
mood_labels = {0: '슬픈', 1: '무거운', 2: '설레는', 3: '신나는', 4: '평화로운'}
df['mood_label'] = df['mood_cluster'].map(mood_labels)

# 추천 함수
def recommend_song_by_emotion(emotion, top_n=5):
    mood = mood_labels[emotion]
    recs = df[df['mood_cluster'] == emotion].sample(n=top_n)[['name', 'artists']]
    return mood, recs

if __name__ == "__main__":
    k = 0
    while k >= 0 and k <= 4:
        k = int(input("검색할 감정을 입력하세요 0-4: "))
        mood, recs = recommend_song_by_emotion(k)
        print(mood)
        print(recs)
    else:
        print("잘못된 입력입니다. 감정은 0-4 사이의 숫자로 입력해주세요.")
