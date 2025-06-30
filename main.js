const imageInput = document.getElementById('imageInput');
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
});
