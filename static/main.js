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

    // 서버 응답이 성공적인지 확인
    if (!response.ok) {
      const errorText = await response.text(); // 오류 메시지 확인
      throw new Error(`서버 오류: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const data = await response.json(); // 서버에서 JSON 받음

    // 키워드 보여주기
    // data.keywords가 존재하고 배열일 때만 map을 사용하도록 수정
    keywordResult.innerHTML = `
      <p class="label">이미지 키워드</p>
      <div class="tags">
        ${data.keywords && Array.isArray(data.keywords) ? data.keywords.map(k => `<span>#${k}</span>`).join('') : '<span>키워드를 불러올 수 없습니다.</span>'}
      </div>
    `;
    keywordResult.style.display = 'block';

    // 추천 음악 보여주기
    musicList.innerHTML = '';
    // data.songs가 존재하고 배열일 때만 forEach를 사용하도록 수정
    if (data.songs && Array.isArray(data.songs) && data.songs.length > 0) {
      data.songs.forEach(song => {
        const li = document.createElement('li');
        li.textContent = `${song.name} - ${song.artists}`;
        musicList.appendChild(li);
      });
    } else {
      const li = document.createElement('li');
      li.textContent = '추천 음악을 찾을 수 없습니다.';
      musicList.appendChild(li);
    }
    musicResult.style.display = 'block';

  } catch (error) {
    console.error('추천 요청 실패:', error);
    alert(`추천 요청 중 오류가 발생했습니다: ${error.message || error}`);
  }
});

// 히스토리 날짜 클릭 시 상세정보 토글
window.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.history-date').forEach(function(dateElem) {
    dateElem.addEventListener('click', function() {
      // 모든 아이템 닫기
      document.querySelectorAll('.history-item').forEach(function(item) {
        item.classList.remove('open');
        const thumb = item.querySelector('.history-thumb');
        const songs = item.querySelector('.history-songs');
        if (thumb) thumb.style.display = 'none';
        if (songs) songs.style.display = 'none';
      });
      // 현재 아이템만 열기
      const parent = this.closest('.history-item');
      if (parent) {
        parent.classList.add('open');
        const thumb = parent.querySelector('.history-thumb');
        const songs = parent.querySelector('.history-songs');
        if (thumb) thumb.style.display = 'flex';
        if (songs) songs.style.display = 'block';
      }
    });
  });
});
