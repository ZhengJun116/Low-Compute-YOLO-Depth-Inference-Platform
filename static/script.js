
const confInput = document.getElementById('conf-input');
const iouInput = document.getElementById('iou-input');
const confVal = document.getElementById('conf-val');
const iouVal = document.getElementById('iou-val');

confInput.addEventListener('input', (e) => { confVal.innerText = e.target.value; });
iouInput.addEventListener('input', (e) => { iouVal.innerText = e.target.value; });

const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const historySelect = document.getElementById('history-select');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const btnSwitch = document.getElementById('btn-switch');
const mediaContainer = document.getElementById('media-container');
const contactsBody = document.getElementById('contacts-body');
const loadingOverlay = document.getElementById('loading-overlay');
const feedStatusText = document.getElementById('feed-status-text');

let currentData = null;
let lastUploadedFile = null;
let isShowingInferred = false;
let isPlaying = true;

async function loadHistory() {
    const res = await fetch('/api/history');
    const history = await res.json();
    historySelect.innerHTML = '<option value="">-- Select Previous Target --</option>';
    history.forEach(item => {
        const option = document.createElement('option');
        option.value = JSON.stringify(item);
        option.text = `[${item.type.toUpperCase()}] ${item.filename}`;
        historySelect.appendChild(option);
    });
}
loadHistory();

dropArea.addEventListener('click', () => fileInput.click());
dropArea.addEventListener('dragover', (e) => { e.preventDefault(); dropArea.style.borderColor = '#52a9a0'; });
dropArea.addEventListener('dragleave', () => { dropArea.style.borderColor = '#1a3340'; });
dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.style.borderColor = '#1a3340';
    if(e.dataTransfer.files.length) {
        lastUploadedFile = e.dataTransfer.files[0];
        handleUpload(lastUploadedFile);
    }
});
fileInput.addEventListener('change', (e) => {
    if(e.target.files.length) {
        lastUploadedFile = e.target.files[0];
        handleUpload(lastUploadedFile);
    }
});

async function handleUpload(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf', confInput.value);
    formData.append('iou', iouInput.value);

    loadingOverlay.style.display = 'flex';
    feedStatusText.innerText = 'UPLOADING & INFERRING...';

    try {
        const res = await fetch('/api/upload_infer', { method: 'POST', body: formData });
        const data = await res.json();
        currentData = data;
        isShowingInferred = true;
        renderMedia();
        renderContacts();
        loadHistory();
        feedStatusText.innerText = 'TRACKING CONFIRMED';
    } catch(err) {
        alert("Error during inference!");
        feedStatusText.innerText = 'ERROR';
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

async function handleReinfer() {
    if(!currentData) return;

    const formData = new FormData();
    formData.append('original_url', currentData.original_url);
    formData.append('conf', confInput.value);
    formData.append('iou', iouInput.value);

    loadingOverlay.style.display = 'flex';
    feedStatusText.innerText = 'RE-INFERRING WITH NEW PARAMS...';

    try {
        const res = await fetch('/api/reinfer', { method: 'POST', body: formData });
        const data = await res.json();
        if(data.error) {
            alert(data.error);
            return;
        }
        data.original_url = currentData.original_url;
        currentData = data;
        isShowingInferred = true;
        renderMedia();
        renderContacts();
        loadHistory();
        feedStatusText.innerText = 'TRACKING CONFIRMED (RE-INFERRED)';
    } catch(err) {
        alert("Error during re-inference!");
        feedStatusText.innerText = 'ERROR';
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

historySelect.addEventListener('change', (e) => {
    if(!e.target.value) return;
    currentData = JSON.parse(e.target.value);
    isShowingInferred = true;
    renderMedia();
    renderContacts();
    feedStatusText.innerText = 'TRACKING (ARCHIVED)';
});

function renderMedia() {
    if(!currentData) return;
    const url = isShowingInferred ? currentData.inferred_url : currentData.original_url;
    const cacheBust = '?t=' + Date.now();

    if(currentData.type === 'video') {
        mediaContainer.innerHTML = `<video id="active-video" src="${url}${cacheBust}" autoplay loop muted></video>`;
        const vid = document.getElementById('active-video');
        if(!isPlaying) vid.pause();
    } else {
        mediaContainer.innerHTML = `<img src="${url}${cacheBust}">`;
    }
}

function renderContacts() {
    if(!currentData) return;
    contactsBody.innerHTML = '';
    currentData.contacts.forEach(c => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${c.track}</td>
            <td>Class ${c.class} [${c.conf}]</td>
            <td class="status-confirmed">${c.status}</td>
            <td>${c.position}</td>
            <td>${c.size}</td>
            <td>${c.distance}</td>
        `;
        contactsBody.appendChild(tr);
    });
}

btnSwitch.addEventListener('click', () => {
    if(!currentData) return;
    isShowingInferred = !isShowingInferred;
    renderMedia();
});

btnStop.addEventListener('click', () => {
    isPlaying = false;
    const vid = document.getElementById('active-video');
    if(vid) vid.pause();
    feedStatusText.innerText = 'PAUSED';
});

btnStart.addEventListener('click', () => {
    if(currentData) {
        handleReinfer();
    }
});
