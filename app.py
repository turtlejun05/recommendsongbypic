from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os
import csv
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 실제 배포시엔 더 복잡하게!

HISTORY_FILE = 'user_history.json'

def read_users():
    users = {}
    if os.path.exists('users.csv'):
        with open('users.csv', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                users[row['username']] = {'password_hash': row['password_hash'], 'nickname': row['nickname']}
    return users

def write_user(username, password_hash, nickname):
    file_exists = os.path.exists('users.csv')
    with open('users.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['username', 'password_hash', 'nickname'])
        writer.writerow([username, password_hash, nickname])

@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        nickname = request.form['nickname']
        users = read_users()
        # 아이디 중복 체크
        if username in users:
            flash('이미 존재하는 아이디입니다.')
            return redirect(url_for('register'))
        # 닉네임 중복 체크
        for u in users.values():
            if u['nickname'] == nickname:
                flash('이미 존재하는 닉네임입니다.')
                return redirect(url_for('register'))
        password_hash = generate_password_hash(password)
        write_user(username, password_hash, nickname)
        flash('회원가입이 완료되었습니다. 로그인 해주세요.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/check_nickname', methods=['POST'])
def check_nickname():
    nickname = request.form['nickname']
    users = read_users()
    for u in users.values():
        if u['nickname'] == nickname:
            return jsonify({'result': 'duplicate'})
    return jsonify({'result': 'ok'})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = read_users()
        if username in users and check_password_hash(users[username]['password_hash'], password):
            session['username'] = username
            session['nickname'] = users[username]['nickname']
            return redirect(url_for('main'))
        else:
            flash('아이디 또는 비밀번호가 올바르지 않습니다.')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('nickname', None)
    return redirect(url_for('login'))

@app.route('/main')
def main():
    username = session.get('username')
    nickname = session.get('nickname')
    history = load_user_history(username) if username else []
    return render_template('main.html', username=username, nickname=nickname, history=history)

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

        # 히스토리 저장
        username = session.get('username')
        if username:
            record = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'image': image_path,
                'songs': songs
            }
            save_user_history(username, record)

        return jsonify({
            "keywords": top_keywords,
            "mood": mood,
            "songs": songs
        })

    except Exception as e:
        print(" 서버 오류 발생:", e)
        return jsonify({"error": str(e)}), 500

def load_user_history(username):
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get(username, [])

def save_user_history(username, record):
    data = {}
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    user_history = data.get(username, [])
    user_history.insert(0, record)  # 최신 기록을 앞에 추가
    user_history = user_history[:10]  # 최대 10개만 유지
    data[username] = user_history
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@app.route('/delete_history', methods=['POST'])
def delete_history():
    username = session.get('username')
    if not username:
        return jsonify({'result': 'error', 'msg': '로그인 필요'}), 401
    index = int(request.form.get('index', -1))
    if index < 0:
        return jsonify({'result': 'error', 'msg': '잘못된 요청'}), 400
    # 히스토리 불러오기
    data = {}
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    user_history = data.get(username, [])
    if index >= len(user_history):
        return jsonify({'result': 'error', 'msg': '존재하지 않는 항목'}), 400
    # 이미지 파일도 삭제(선택)
    image_path = user_history[index].get('image')
    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
        except Exception:
            pass
    # 항목 삭제
    del user_history[index]
    data[username] = user_history
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return jsonify({'result': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
