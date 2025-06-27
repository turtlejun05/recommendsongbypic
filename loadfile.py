import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

# Load dataset
df = pd.read_csv("data.csv")

# Select relevant features
features = df[['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness', 'liveness']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Autoencoder 구성
input_dim = X_scaled.shape[1]
encoding_dim = 3

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='linear')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)

autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=256, shuffle=True, verbose=0)

# 인코딩된 특징 추출
X_encoded = encoder.predict(X_scaled)

# Train KMeans model (on encoded data)
kmeans = KMeans(n_clusters=6, random_state=42)
df['mood_cluster'] = kmeans.fit_predict(X_encoded)

# Map clusters to mood labels
mood_labels = {
    0: '신나는',
    1: '슬픈',
    2: '감성적인',
    3: '몽환적인',
    4: '잔잔한',
    5: '무거운'
}
df['mood_label'] = df['mood_cluster'].map(mood_labels)

# Function to recommend songs
def recommend_song(valence, energy, danceability, acousticness, instrumentalness, liveness, top_n=5):
    user_input = np.array([[valence, energy, danceability, acousticness, instrumentalness, liveness]])
    user_scaled = scaler.transform(user_input)
    user_encoded = encoder.predict(user_scaled)
    cluster = kmeans.predict(user_encoded)[0]
    mood = mood_labels[cluster]
    recs = df[df['mood_cluster'] == cluster].sample(n=top_n)[['name', 'artists', 'mood_label']]
    return mood, recs

if __name__ == "__main__":
    print("🎵 분위기에 따른 음악 추천기 🎵")
    # 전체 실루엣 계수 출력
    score = silhouette_score(X_encoded, df['mood_cluster'])
    print(f"📊 전체 실루엣 계수 (평균): {score:.3f}")

    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_encoded)
        score = silhouette_score(X_encoded, labels)
        print(f"k={k} → 실루엣 계수={score:.3f}")
    # valence = float(input("valence (0~1): "))
    # energy = float(input("energy (0~1): "))
    # danceability = float(input("danceability (0~1): "))
    # acousticness = float(input("acousticness (0~1): "))
    # instrumentalness = float(input("instrumentalness (0~1): "))
    # liveness = float(input("liveness (0~1): "))

    # mood, recommendations = recommend_song(
    #     valence, energy, danceability, acousticness, instrumentalness, liveness
    # )

    # print(f"\n🎧 예측된 분위기: {mood}")
    # print("🎵 추천된 노래:")
    # print(recommendations.to_string(index=False))

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
        '감성적인': 'orange',
        '몽환적인': 'purple',
        '잔잔한': 'green',
        '무거운': 'gray'
    }

    import matplotlib
    matplotlib.rc('font', family='AppleGothic')  # macOS용 한글 폰트 설정
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