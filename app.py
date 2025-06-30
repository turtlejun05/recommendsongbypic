from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        uploaded_file = request.files.get('image')
        if not uploaded_file:
            return jsonify({"error": "이미지를 업로드해주세요."}), 400

        os.makedirs('static/uploads', exist_ok=True)
        image_path = os.path.join('static/uploads', uploaded_file.filename)
        uploaded_file.save(image_path)

        from clip_model import get_top_emotion
        top_keywords = get_top_emotion(image_path)

        from music_model import recommend_song_by_emotion
        mood, song_df = recommend_song_by_emotion(top_keywords[0])
        songs = song_df.to_dict(orient='records')

        return jsonify({
            "keywords": top_keywords,
            "mood": mood,
            "songs": songs
        })

    except Exception as e:
        print(" 서버 오류 발생:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
