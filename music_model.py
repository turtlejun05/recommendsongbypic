import pandas as pd
import numpy as np
import random
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

# # Autoencoder
# input_dim = X_scaled.shape[1]
# encoding_dim = 3
# input_layer = Input(shape=(input_dim,))
# encoded = Dense(encoding_dim, activation='relu')(input_layer)
# decoded = Dense(input_dim, activation='linear')(encoded)
# autoencoder = Model(inputs=input_layer, outputs=decoded)
# encoder = Model(inputs=input_layer, outputs=encoded)
# autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
# autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=512, shuffle=True, verbose=0)

# # 인코딩 & KMeans
# X_encoded = encoder.predict(X_scaled)
kmeans = KMeans(n_clusters=5, random_state=42)
df['mood_cluster'] = kmeans.fit_predict(X_scaled)

# 감정 매핑
mood_labels = {0: '신나는', 1: '슬픈', 2: '설레는', 3: '평화로운', 4: '무거운'}
df['mood_label'] = df['mood_cluster'].map(mood_labels)

# 감정명 -> mood_classifiyer 번호
emotion_to_id = {'신나는':0, '슬픈':1, '설레는':2, '평화로운':3, '무거운':4}

mood_classifiyer = {
    0 : [0.9, 0.9, 0.8, random.uniform(0.1, 0.9), random.uniform(0.1, 0.7), 0.9, 0.8],
    1 : [0.2, 0.3, 0.5, 0.8, random.uniform(0.1, 0.7), 0.2, random.uniform(0.2, 0.7)],
    2 : [0.7, 0.6, 0.6, random.uniform(0.1, 0.9), random.uniform(0.1, 0.7), 0.8, 0.7],
    3 : [0.6, 0.2, 0.3, 0.8, random.uniform(0.1, 0.7), 0.6, random.uniform(0.2, 0.7)],
    4 : [random.uniform(0.1, 0.7), 0.9, 0.8, random.uniform(0.1, 0.9), 0.6, 0.2, 0.5],
}

def recommend_song_by_emotion(emotion, top_n=5):
    mood_id = emotion_to_id.get(emotion)
    if mood_id is None:
        return None, None
    danceability, energy, loudness, acousticness, instrumentalness, valence, tempo = mood_classifiyer[mood_id]
    user_input = np.array([[danceability, energy, loudness, acousticness, instrumentalness, valence, tempo]])
    user_scaled = scaler.transform(user_input)
    user_encoded = encoder.predict(user_scaled)
    cluster = kmeans.predict(user_encoded)[0]
    mood = mood_labels[cluster]
    recs = df[df['mood_cluster'] == cluster].sample(n=top_n)[['name', 'artists', 'mood_label']]
    return mood, recs

if __name__ == "__main__":
    k = input("검색할 감정을 입력하세요 0-4: ")
    if k.isdigit() and 0 <= int(k) <= 4:
        mood, recs = recommend_song_by_emotion(mood_labels[int(k)])
        print(mood)
        print(recs)
    else:
        print("잘못된 입력입니다. 감정은 0-4 사이의 숫자로 입력해주세요.")
