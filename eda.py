import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv("data.csv")

# 주요 수치형 변수 선택
numeric_cols = ['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness',
                'liveness', 'speechiness', 'loudness', 'tempo', 'duration_ms']

# 상관행렬 계산
corr = df[numeric_cols].corr()

# 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("음악 특성 간 상관관계 히트맵")
plt.tight_layout()
plt.show()