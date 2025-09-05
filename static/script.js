const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const emotionText = document.getElementById('emotion');

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
    video.play();
});

function sendFrame(){
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg');
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
    })
    .then(response => response.json())
    .then(data => {
        emotionText.innerText = data.emotion;
    });
}

setInterval(sendFrame, 1500); // Send frame every 1.5 seconds
