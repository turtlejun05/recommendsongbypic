import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

# Spotify API 인증 정보
client_id = '95a02d14c9364bbba1688c3fa9893b90'
client_secret = '1c266ad783e44409b3c722cc878506b1'

# 인증 설정
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# 수집할 장르 또는 검색어 목록
search_queries = ['chill', 'sad', 'happy', 'rock', 'jazz', 'electronic']

# 최종 저장할 데이터 리스트
track_data = []

# 각 키워드에 대해 검색 & 오디오 피처 수집
for query in search_queries:
    print(f"🔍 Searching for: {query}")
    results = sp.search(q=query, type='track', limit=50)  # 최대 50개
    tracks = results['tracks']['items']
    
    for track in tracks:
        track_id = track['id']
        track_name = track['name']
        artist_name = track['artists'][0]['name']
        album_name = track['album']['name']
        release_date = track['album']['release_date']
        
        try:
            features = sp.audio_features([track_id])[0]
        except:
            continue  # 실패하면 skip
        
        if features:
            track_info = {
                'id': track_id,
                'name': track_name,
                'artist': artist_name,
                'album': album_name,
                'release_date': release_date,
                'danceability': features['danceability'],
                'energy': features['energy'],
                'loudness': features['loudness'],
                'acousticness': features['acousticness'],
                'instrumentalness': features['instrumentalness'],
                'valence': features['valence'],
                'tempo': features['tempo'],
                'query': query
            }
            track_data.append(track_info)
        
        time.sleep(0.1)  # rate limit 대응

# DataFrame으로 변환 & CSV 저장
df = pd.DataFrame(track_data)
df.to_csv('spotify_dataset.csv', index=False, encoding='utf-8-sig')
print(f"✅ {len(df)}곡 저장 완료 → 'spotify_dataset.csv'")