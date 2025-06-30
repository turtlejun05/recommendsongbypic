/*const imageInput = document.getElementById('imageInput');
const imagePreviewContainer = document.getElementById('imagePreviewContainer');
const uploadBox = document.getElementById('uploadBox');
const recommendButton = document.getElementById('recommendButton');
const keywordResult = document.getElementById('keywordResult');
const musicResult = document.getElementById('musicResult');
const musicList = document.getElementById('musicList');

imageInput.addEventListener('change', function () {
  const files = Array.from(this.files);
  imagePreviewContainer.innerHTML = '';

  if (files.length > 0) {
    uploadBox.style.display = 'none';

    files.slice(0, 4).forEach(file => {
      const reader = new FileReader();
      reader.onload = function (e) {
        const img = document.createElement('img');
        img.src = e.target.result;
        imagePreviewContainer.appendChild(img);
      };
      reader.readAsDataURL(file);
    });

    imagePreviewContainer.style.display = 'flex';
    recommendButton.style.display = 'inline-block';
    keywordResult.style.display = 'none';
    musicResult.style.display = 'none';
  }
});

recommendButton.addEventListener('click', function () {
  keywordResult.style.display = 'block';

  setTimeout(() => {
    musicList.innerHTML = '';
    const dummySongs = [
      'Olive Breeze - Acoustic Vibe',
      'Forest Chill - Lo-fi Instrumental',
      'Morning Mist - Piano Mood',
      'Nature Flow - Ambient Sound'
    ];

    dummySongs.forEach(song => {
      const li = document.createElement('li');
      li.textContent = song;
      musicList.appendChild(li);
    });

    musicResult.style.display = 'block';
  }, 500);
});*/
const imageInput = document.getElementById('imageInput');
const imagePreviewContainer = document.getElementById('imagePreviewContainer');
const uploadBox = document.getElementById('uploadBox');
const recommendButton = document.getElementById('recommendButton');
const keywordResult = document.getElementById('keywordResult');
const musicResult = document.getElementById('musicResult');
const musicList = document.getElementById('musicList');

// 이미지 선택 시 미리보기
imageInput.addEventListener('change', function () {
  const files = Array.from(this.files);
  imagePreviewContainer.innerHTML = '';

  if (files.length > 0) {
    uploadBox.style.display = 'none';

    files.slice(0, 4).forEach(file => {
      const reader = new FileReader();
      reader.onload = function (e) {
        const img = document.createElement('img');
        img.src = e.target.result;
        imagePreviewContainer.appendChild(img);
      };
      reader.readAsDataURL(file);
    });

    imagePreviewContainer.style.display = 'flex';
    recommendButton.style.display = 'inline-block';
    keywordResult.style.display = 'none';
    musicResult.style.display = 'none';
  }
});

// 버튼 클릭 시 서버에 이미지 보내고 결과 받아서 출력
recommendButton.addEventListener('click', async function () {
  const file = imageInput.files[0];
  if (!file) return alert('이미지를 선택해주세요.');

  // 서버에 보낼 FormData 생성
  const formData = new FormData();
  formData.append('image', file);

  try {
    // POST 요청
    const response = await fetch('/recommend', {
      method: 'POST',
      body: formData
    });

    const data = await response.json(); // 서버에서 JSON 받음

    // 키워드 보여주기
    keywordResult.innerHTML = `
      <p class="label">이미지 키워드</p>
      <div class="tags">
        ${data.keywords.map(k => `<span>#${k}</span>`).join('')}
      </div>
    `;
    keywordResult.style.display = 'block';

    // 추천 음악 보여주기
    musicList.innerHTML = '';
    data.songs.forEach(song => {
      const li = document.createElement('li');
      li.textContent = `${song.name} - ${song.artists}`;
      musicList.appendChild(li);
    });
    musicResult.style.display = 'block';

  } catch (error) {
    console.error('추천 요청 실패:', error);
    alert('추천 요청 중 오류가 발생했습니다.');
  }
});
