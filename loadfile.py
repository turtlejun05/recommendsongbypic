import pandas as pd
import numpy as np
import random

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

# Load dataset
url = 'https://raw.githubusercontent.com/turtlejun05/recommendsongbypic/refs/heads/main/data.csv'
df = pd.read_csv(url)

# Select relevant features
features = df[['danceability','energy', 'loudness','acousticness','instrumentalness','valence', 'tempo']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# GPU 확인 및 메모리 설정
print("GPU 사용 가능 여부:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 메모리 성장 설정 완료")
    except RuntimeError as e:
        print(e)

# Autoencoder 구성
input_dim = X_scaled.shape[1]
encoding_dim = 3

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='linear')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)

autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=512, shuffle=True, verbose=0)

# 인코딩된 특징 추출
X_encoded = encoder.predict(X_scaled)

# Train KMeans model (on encoded data)
kmeans = KMeans(n_clusters=5, random_state=42)
df['mood_cluster'] = kmeans.fit_predict(X_encoded)

# Map clusters to mood labels
# 요기 clip 이랑 연결 시켜야함
mood_labels = {
    0: '신나는',
    1: '슬픈',
    2: '설레는',
    3: '평화로운',
    4: '무거운',
}
df['mood_label'] = df['mood_cluster'].map(mood_labels)

mood_classifiyer = {
    0 : [0.9, 0.9, 0.8, random.uniform(0.1, 0.9), random.uniform(0.1, 0.7), 0.9, 0.8],
    1 : [0.2, 0.3, 0.5, 0.8, random.uniform(0.1, 0.7), 0.2, random.uniform(0.2, 0.7)],
    2 : [0.7, 0.6, 0.6, random.uniform(0.1, 0.9), random.uniform(0.1, 0.7), 0.8, 0.7],
    3 : [0.6, 0.2, 0.3, 0.8, random.uniform(0.1, 0.7), 0.6, random.uniform(0.2, 0.7)],
    4 : [random.uniform(0.1, 0.7), 0.9, 0.8, random.uniform(0.1, 0.9), 0.6, 0.2, 0.5],
}

# Function to recommend songs
def recommend_song(danceability, energy, loudness, acousticness, instrumentalness, valence, tempo, top_n=5):
    user_input = np.array([[danceability, energy, loudness, acousticness, instrumentalness, valence, tempo]])
    user_scaled = scaler.transform(user_input)
    user_encoded = encoder.predict(user_scaled)
    cluster = kmeans.predict(user_encoded)[0]
    mood = mood_labels[cluster]
    recs = df[df['mood_cluster'] == cluster].sample(n=top_n)[['name', 'artists', 'mood_label']]
    return mood, recs

if __name__ == "__main__":
    print("분위기에 따른 음악 추천기")
    k = int(input("추천할 노래의 타입"))
    if k < 0 or k > 5:
        print("잘못된 입력입니다. 0에서 5 사이의 숫자를 입력하세요.")
        exit()
    danceability = mood_classifiyer[k][0]
    energy = mood_classifiyer[k][1]
    loudness = mood_classifiyer[k][2]
    acousticness = mood_classifiyer[k][3]
    instrumentalness = mood_classifiyer[k][4]
    valence = mood_classifiyer[k][5]
    tempo = mood_classifiyer[k][6]

    mood, recommendations = recommend_song(
        danceability,energy, loudness,acousticness,instrumentalness,valence, tempo
    )

    print(f"\n 예측된 분위기: {mood}")
    print("추천된 노래:")
    print(recommendations.to_string(index=False))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_clusters():
    # PCA로 2차원 축소
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df_plot = pd.DataFrame(components, columns=['x', 'y'])
    df_plot['mood_label'] = df['mood_label']

    # 색상 매핑
    colors = {
        '신나는': 'red',
        '슬픈': 'blue',
        '설레는': 'orange',
        '평화로운': 'purple',
        '무거운': 'green',
    }

    import matplotlib
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

    plt.figure(figsize=(10, 6))
    for label in df_plot['mood_label'].unique():
        subset = df_plot[df_plot['mood_label'] == label]
        plt.scatter(subset['x'], subset['y'], c=colors[label], label=label, alpha=0.6)

    plt.title('KMeans 클러스터 시각화 (mood_label 기준)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 사용 예시 (원할 때 호출)
plot_clusters()