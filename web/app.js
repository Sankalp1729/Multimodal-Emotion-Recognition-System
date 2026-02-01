(function(){
  const imageChooseBtn = document.getElementById('image-choose');
  const imageFileInput = document.getElementById('image-file');
  const imageDrop = document.getElementById('image-drop');
  const imagePreview = document.getElementById('image-preview');
  const imageThumb = document.getElementById('image-thumb');
  const imageMeta = document.getElementById('image-meta');
  const imageClearBtn = document.getElementById('image-clear');

  const audioFileInput = document.getElementById('audio-file');
  const audioMeta = document.getElementById('audio-meta');
  const audioClearBtn = document.getElementById('audio-clear');

  const textArea = document.getElementById('text-area');
  const textCount = document.getElementById('text-count');
  const textClearBtn = document.getElementById('text-clear');

  const analyzeBtn = document.getElementById('analyze-btn');
  const resetBtn = document.getElementById('reset-btn');
  const errorBanner = document.getElementById('error-banner');

  const results = document.getElementById('results');
  const finalEmotion = document.getElementById('final-emotion');
  const finalConfidence = document.getElementById('final-confidence');
  const modImageEmotion = document.getElementById('mod-image-emotion');
  const modImageConfidence = document.getElementById('mod-image-confidence');
  const modImageBar = document.getElementById('mod-image-bar');
  const modAudioEmotion = document.getElementById('mod-audio-emotion');
  const modAudioConfidence = document.getElementById('mod-audio-confidence');
  const modAudioBar = document.getElementById('mod-audio-bar');
  const modTextEmotion = document.getElementById('mod-text-emotion');
  const modTextConfidence = document.getElementById('mod-text-confidence');
  const modTextBar = document.getElementById('mod-text-bar');

  // Helpers
  function showError(msg){
    errorBanner.textContent = msg;
    errorBanner.classList.remove('hidden');
    errorBanner.classList.add('visible');
  }
  function clearError(){
    errorBanner.textContent = '';
    errorBanner.classList.remove('visible');
    errorBanner.classList.add('hidden');
  }
  function humanSize(bytes){
    if (!bytes && bytes !== 0) return '';
    const units = ['B','KB','MB','GB'];
    let i = 0; let n = bytes;
    while(n >= 1024 && i < units.length-1){ n /= 1024; i++; }
    return `${n.toFixed(1)} ${units[i]}`;
  }
  function updateAnalyzeEnabled(){
    const hasImage = !!imageFileInput.files[0];
    const hasAudio = !!audioFileInput.files[0];
    const hasText = textArea.value.trim().length > 0;
    analyzeBtn.disabled = !(hasImage || hasAudio || hasText);
  }
  function setLoading(isLoading){
    const spinner = analyzeBtn.querySelector('.spinner');
    const label = analyzeBtn.querySelector('.btn-label');
    if(isLoading){
      analyzeBtn.disabled = true;
      spinner.classList.remove('hidden');
      label.textContent = 'Analyzing…';
    } else {
      spinner.classList.add('hidden');
      label.textContent = 'Analyze Emotion';
      updateAnalyzeEnabled();
    }
  }
  function resetUI(){
    imageFileInput.value = '';
    audioFileInput.value = '';
    textArea.value = '';
    imagePreview.classList.add('hidden');
    audioMeta.textContent = '';
    audioClearBtn.classList.add('hidden');
    textClearBtn.classList.add('hidden');
    textCount.textContent = '0';
    results.classList.add('hidden');
    finalEmotion.textContent = '—';
    finalConfidence.textContent = '—';
    modImageEmotion.textContent = 'Not provided';
    modImageConfidence.textContent = '—';
    modImageBar.style.width = '0%';
    modAudioEmotion.textContent = 'Not provided';
    modAudioConfidence.textContent = '—';
    modAudioBar.style.width = '0%';
    modTextEmotion.textContent = 'Not provided';
    modTextConfidence.textContent = '—';
    modTextBar.style.width = '0%';
    clearError();
    updateAnalyzeEnabled();
  }

  // Image input
  imageChooseBtn.addEventListener('click', ()=> imageFileInput.click());
  imageDrop.addEventListener('dragover', (e)=>{ e.preventDefault(); });
  imageDrop.addEventListener('drop', (e)=>{
    e.preventDefault();
    if(e.dataTransfer.files && e.dataTransfer.files[0]){
      imageFileInput.files = e.dataTransfer.files;
      handleImageChange();
    }
  });
  imageDrop.addEventListener('click', ()=> imageFileInput.click());

  function handleImageChange(){
    const file = imageFileInput.files[0];
    if(!file){ updateAnalyzeEnabled(); return; }
    const url = URL.createObjectURL(file);
    imageThumb.src = url;
    imagePreview.classList.remove('hidden');
    imageMeta.textContent = `${file.name} • ${humanSize(file.size)}`;
    updateAnalyzeEnabled();
  }
  imageFileInput.addEventListener('change', handleImageChange);
  imageClearBtn.addEventListener('click', ()=>{
    imageFileInput.value = '';
    imagePreview.classList.add('hidden');
    updateAnalyzeEnabled();
  });

  // Audio input
  audioFileInput.addEventListener('change', ()=>{
    const file = audioFileInput.files[0];
    if(file){
      audioMeta.textContent = `${file.name} • ${humanSize(file.size)}`;
      audioClearBtn.classList.remove('hidden');
    } else {
      audioMeta.textContent = '';
      audioClearBtn.classList.add('hidden');
    }
    updateAnalyzeEnabled();
  });
  audioClearBtn.addEventListener('click', ()=>{
    audioFileInput.value = '';
    audioMeta.textContent = '';
    audioClearBtn.classList.add('hidden');
    updateAnalyzeEnabled();
  });

  // Text input
  textArea.addEventListener('input', ()=>{
    const len = textArea.value.length;
    textCount.textContent = len;
    if(len > 0){ textClearBtn.classList.remove('hidden'); } else { textClearBtn.classList.add('hidden'); }
    updateAnalyzeEnabled();
  });
  textClearBtn.addEventListener('click', ()=>{
    textArea.value = '';
    textCount.textContent = '0';
    textClearBtn.classList.add('hidden');
    updateAnalyzeEnabled();
  });

  // Analyze handler
  analyzeBtn.addEventListener('click', async ()=>{
    clearError();
    setLoading(true);
    try {
      const formData = new FormData();
      if(imageFileInput.files[0]) formData.append('image_file', imageFileInput.files[0]);
      if(audioFileInput.files[0]) formData.append('audio_file', audioFileInput.files[0]);
      const textVal = textArea.value.trim();
      if(textVal) formData.append('text', textVal);

      const resp = await fetch('/predict', { method: 'POST', body: formData });
      if(!resp.ok){
        const text = await resp.text();
        throw new Error(`Server error (${resp.status}): ${text}`);
      }
      const data = await resp.json();
      // Expected shape: { emotion, confidence, details: { per_modality: { image: {...}, audio: {...}, text: {...} } } } }
      finalEmotion.textContent = (data.emotion || '—').toUpperCase();
      const confPct = (data.confidence != null) ? (data.confidence * 100).toFixed(1) + '%' : '—';
      finalConfidence.textContent = confPct;

      const mods = (data.details && (data.details.per_modality || data.details.modalities)) ? (data.details.per_modality || data.details.modalities) : {};

      function topEmotionAndConf(probDist){
        if(!probDist || typeof probDist !== 'object') return { emotion: null, confidence: null };
        // If backend provided emotion/confidence fields, use them.
        if('emotion' in probDist || 'confidence' in probDist){
          const e = probDist.emotion || null;
          const c = (probDist.confidence != null) ? probDist.confidence : null;
          return { emotion: e, confidence: c };
        }
        // Otherwise compute from distribution by argmax
        let topE = null; let topV = -1;
        for(const [k,v] of Object.entries(probDist)){
          const val = typeof v === 'number' ? v : parseFloat(v);
          if(!isNaN(val) && val > topV){ topV = val; topE = k; }
        }
        if(topE === null) return { emotion: null, confidence: null };
        return { emotion: topE, confidence: topV };
      }

      function fillMod(modKey, elEmotion, elConf, elBar){
        const m = mods && mods[modKey];
        if(!m){
          elEmotion.textContent = 'Not provided';
          elConf.textContent = '—';
          elBar.style.width = '0%';
          return;
        }
        const { emotion, confidence } = topEmotionAndConf(m);
        const e = emotion || '—';
        const cPct = (confidence != null) ? (confidence * 100).toFixed(1) + '%' : '—';
        elEmotion.textContent = e.toUpperCase();
        elConf.textContent = cPct;
        elBar.style.width = (confidence != null) ? `${(confidence * 100).toFixed(0)}%` : '0%';
      }

      fillMod('image', modImageEmotion, modImageConfidence, modImageBar);
      fillMod('audio', modAudioEmotion, modAudioConfidence, modAudioBar);
      fillMod('text', modTextEmotion, modTextConfidence, modTextBar);

      results.classList.remove('hidden');
    } catch(err){
      console.error(err);
      showError(err.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  });

  // Reset
  resetBtn.addEventListener('click', resetUI);

  // Init
  resetUI();
})();