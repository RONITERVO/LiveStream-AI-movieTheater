(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const player = $('player');
  const activeStatuses = new Set(['starting', 'planning', 'generating', 'narrating', 'aligning', 'buffering', 'running']);
  let state = null;
  let activeId = null;
  let activeMode = 'story';
  let activeTab = 'scenes';
  let currentIndex = -1;
  let pollTimer = null;
  let waitingForNext = false;
  let directiveScope = 'next_scene';
  let directiveDelivery = 'after_buffer';
  let directiveBusy = false;
  let muted = false;
  let captions = true;
  let languageChoices = [];
  let defaultTranslationLanguage = 'fi';
  const highlighter = new window.CaptionHighlighter($('captionText'), (seconds) => {
    player.currentTime = Math.max(0, Math.min(Number(seconds) || 0, player.duration || Number(seconds) || 0));
  });
  const interlude = new window.LearningInterlude({
    shell: $('playerShell'), window: $('captionWindow'), transcript: $('captionText'),
    root: $('learningInterlude'), pair: $('learningPair'), source: $('learningSource'),
    translation: $('learningTranslation'), meta: $('learningMeta'),
  });

  function toast(message) {
    const target = $('toast');
    target.textContent = String(message || '');
    target.classList.add('show');
    clearTimeout(target._timer);
    target._timer = setTimeout(() => target.classList.remove('show'), 2800);
  }

  function formatTime(seconds) {
    const value = Math.max(0, Math.round(Number(seconds) || 0));
    const minutes = Math.floor(value / 60);
    return `${minutes}:${String(value % 60).padStart(2, '0')}`;
  }

  function mediaUrl(path) {
    return `/api/video?path=${encodeURIComponent(path)}`;
  }

  function setMode(mode) {
    activeMode = mode;
    $('modeSwitch').querySelectorAll('button').forEach((button) => {
      button.classList.toggle('active', button.dataset.mode === mode);
    });
    const pasted = mode === 'my_story';
    $('promptInput').hidden = pasted;
    $('promptInput').required = !pasted;
    $('storyTextInput').hidden = !pasted;
    $('storyTextInput').required = pasted;
    (pasted ? $('storyTextInput') : $('promptInput')).focus();
  }

  function appendOption(select, value, label, selected) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    option.selected = Boolean(selected);
    select.append(option);
  }

  function availableTranslation(source, preferred) {
    if (preferred && preferred !== source) return preferred;
    if (defaultTranslationLanguage !== source) return defaultTranslationLanguage;
    if (source !== 'en') return 'en';
    return languageChoices.find((item) => item.translation && item.value !== source)?.value || 'en';
  }

  function showFlag(flagId, pickerId, language, prefix) {
    const choice = languageChoices.find((item) => item.value === language);
    $(flagId).querySelector('use').setAttribute('href', `/static/flags.svg#${choice?.flag || `flag-${language}`}`);
    $(pickerId).title = `${prefix}: ${choice?.label || language.toUpperCase()}`;
  }

  function starterAttention(detail = 'Starting local models') {
    const source = languageChoices.find((item) => item.value === $('languageSelect').value);
    const target = languageChoices.find((item) => item.value === $('translationSelect').value);
    return {
      pairs: [{ source: source?.starter || 'story', translation: target?.starter || 'story' }],
      detail,
      eta_seconds: null,
    };
  }

  function syncCaptionVisibility() {
    if (interlude.active) {
      $('captionWindow').hidden = false;
      return;
    }
    $('captionWindow').hidden = !captions || currentIndex < 0;
  }

  function syncLanguageControls() {
    const source = $('languageSelect').value || 'en';
    const target = availableTranslation(source, $('translationSelect').value);
    $('languageSelect').value = source;
    $('quickLanguageSelect').value = source;
    $('translationSelect').value = target;
    $('quickTranslationSelect').value = target;
    showFlag('quickLanguageFlag', 'quickLanguagePicker', source, 'Narration language');
    showFlag('quickTranslationFlag', 'quickTranslationPicker', target, 'Translation language');
    [$('translationSelect'), $('quickTranslationSelect')].forEach((select) => {
      [...select.options].forEach((option) => { option.disabled = option.value === source; });
    });
  }

  async function loadOptions() {
    const response = await fetch('/api/config');
    const options = await response.json();
    languageChoices = options.languages || [];
    defaultTranslationLanguage = options.default_translation_language || 'fi';
    options.modes.forEach((mode) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.mode = mode.value;
      button.textContent = mode.label;
      button.addEventListener('click', () => setMode(mode.value));
      $('modeSwitch').append(button);
    });
    options.audiences.forEach((item) => appendOption($('audienceSelect'), item.value, item.label, item.value === 'family'));
    languageChoices.forEach((item) => {
      const quickLabel = `${item.value.toUpperCase()} · ${item.label}`;
      appendOption($('languageSelect'), item.value, item.label, item.value === 'en');
      appendOption($('quickLanguageSelect'), item.value, quickLabel, item.value === 'en');
      if (item.translation) {
        const selected = item.value === defaultTranslationLanguage;
        appendOption($('translationSelect'), item.value, item.label, selected);
        appendOption($('quickTranslationSelect'), item.value, quickLabel, selected);
      }
    });
    options.voices.forEach((voice) => {
      const label = `${voice} (${voice.startsWith('F') ? 'Female' : 'Male'})`;
      appendOption($('voiceSelect'), voice, label, voice === 'M1');
    });
    Object.entries(options.quality || {}).forEach(([key, value]) => {
      const fields = {
        width: 'qualityWidth', height: 'qualityHeight', frames: 'qualityFrames',
        fps: 'qualityFps', min_words: 'minWords', max_words: 'maxWords', max_slow: 'maxSlow',
      };
      if (fields[key]) $(fields[key]).value = value;
    });
    syncLanguageControls();
    setMode('story');
  }

  function requestPayload() {
    return {
      mode: activeMode,
      prompt: $('promptInput').value.trim(),
      story_text: activeMode === 'my_story' ? $('storyTextInput').value : '',
      audience: $('audienceSelect').value,
      language: $('languageSelect').value,
      translation_language: $('translationSelect').value,
      voice: $('voiceSelect').value,
      quality_settings: {
        width: Number($('qualityWidth').value),
        height: Number($('qualityHeight').value),
        frames: Number($('qualityFrames').value),
        fps: Number($('qualityFps').value),
        min_words: Number($('minWords').value),
        max_words: Number($('maxWords').value),
        max_slow: Number($('maxSlow').value),
      },
      context_compaction_scenes: Number($('compactionScenes').value),
      seed: Number($('seedInput').value),
    };
  }

  async function startStory(event) {
    event.preventDefault();
    const button = $('btnStart');
    button.disabled = true;
    interlude.show(starterAttention());
    try {
      const response = await fetch('/api/theater', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload()),
      });
      const next = await response.json();
      if (!response.ok) throw new Error(next.error || 'Could not start story');
      currentIndex = -1;
      waitingForNext = false;
      player.removeAttribute('src');
      player.load();
      renderState(next);
    } catch (error) {
      interlude.hide(captions && currentIndex >= 0);
      toast(error.message);
    } finally {
      button.disabled = false;
    }
  }

  function showScene(index) {
    const scene = state?.segments?.[index];
    if (!scene) return;
    currentIndex = index;
    $('sceneNumber').textContent = `Scene ${scene.number}`;
    $('sceneTitle').textContent = scene.translated_title ? `${scene.title} · ${scene.translated_title}` : scene.title;
    highlighter.render(scene, Number(scene.duration) || player.duration || 1);
    syncCaptionVisibility();
    renderSide();
  }

  async function playScene(index, autoplay = true) {
    const scene = state?.segments?.[index];
    if (!scene) return false;
    if (player.dataset.scene !== String(scene.number)) {
      player.src = mediaUrl(scene.path);
      player.dataset.scene = String(scene.number);
      player.load();
    }
    showScene(index);
    waitingForNext = false;
    interlude.hide(captions);
    if (!autoplay) return true;
    try {
      await player.play();
      return true;
    } catch {
      return false;
    }
  }

  async function advance() {
    if (state?.segments?.[currentIndex + 1]) {
      await playScene(currentIndex + 1);
      return;
    }
    waitingForNext = true;
    if (activeStatuses.has(state?.status)) interlude.show(state?.attention || starterAttention('Preparing next scene'));
  }

  function renderScenes() {
    const body = $('sideBody');
    body.replaceChildren();
    if (!state) return;
    const scenes = state?.segments || [];
    if (!scenes.length) {
      const empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.textContent = activeStatuses.has(state?.status) ? '' : state?.message || '';
      body.append(empty);
      return;
    }
    scenes.forEach((scene, index) => {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = `scene-card${index === currentIndex ? ' active' : ''}`;
      const number = document.createElement('span');
      number.className = 'scene-index';
      number.textContent = String(scene.number);
      const meta = document.createElement('span');
      meta.className = 'scene-meta';
      const title = document.createElement('strong');
      title.textContent = scene.title;
      const detail = document.createElement('small');
      detail.textContent = scene.alignment_model || 'Wan 2.2';
      meta.append(title, detail);
      const duration = document.createElement('span');
      duration.className = 'scene-duration';
      duration.textContent = formatTime(scene.duration);
      card.append(number, meta, duration);
      card.addEventListener('click', () => playScene(index));
      body.append(card);
    });
  }

  function chip(label, value, current, onSelect) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.classList.toggle('active', value === current);
    button.addEventListener('click', () => onSelect(value));
    return button;
  }

  function renderDirect() {
    const body = $('sideBody');
    body.replaceChildren();
    if (!state || !activeId) {
      const empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.textContent = 'Live Chat / Direct';
      body.append(empty);
      return;
    }
    const interactive = state.config?.mode === 'interactive';
    if (!interactive && directiveScope === 'audience_message') directiveScope = 'next_scene';
    const panel = document.createElement('div');
    panel.className = 'direct-panel';
    const scopes = document.createElement('div');
    scopes.className = 'chip-row';
    if (interactive) scopes.append(chip('Chat Host', 'audience_message', directiveScope, (value) => { directiveScope = value; renderDirect(); }));
    scopes.append(
      chip('Event', 'next_scene', directiveScope, (value) => { directiveScope = value; renderDirect(); }),
      chip('Rule', 'persistent', directiveScope, (value) => { directiveScope = value; renderDirect(); }),
    );
    const delivery = document.createElement('div');
    delivery.className = 'chip-row';
    delivery.append(
      chip('Buffer', 'after_buffer', directiveDelivery, (value) => { directiveDelivery = value; renderDirect(); }),
      chip('Fast', 'next_unrendered', directiveDelivery, (value) => { directiveDelivery = value; renderDirect(); }),
    );
    const list = document.createElement('div');
    list.className = 'directive-list';
    (state.live_directives || []).filter((item) => ['pending', 'active'].includes(item.status)).forEach((item) => {
      const card = document.createElement('div');
      card.className = 'directive';
      const label = document.createElement('small');
      label.textContent = item.scope === 'audience_message' ? 'Chat' : item.scope === 'persistent' ? 'Rule' : 'Event';
      const text = document.createElement('p');
      text.textContent = item.text;
      card.append(label, text);
      if (item.scope === 'persistent' && item.status === 'active') {
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'icon-button';
        remove.textContent = '×';
        remove.addEventListener('click', () => removeDirective(item.id));
        card.append(remove);
      }
      list.append(card);
    });
    const composer = document.createElement('div');
    composer.className = 'direct-composer';
    const input = document.createElement('textarea');
    input.id = 'directiveInput';
    input.placeholder = directiveScope === 'audience_message'
      ? 'Ask host a question or speak…'
      : directiveScope === 'persistent' ? 'Add lasting world rule…' : 'Direct next scene event…';
    const send = document.createElement('button');
    send.type = 'button';
    send.className = 'start-button';
    send.textContent = 'Send';
    send.addEventListener('click', () => sendDirective(input, send));
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendDirective(input, send); }
    });
    composer.append(input, send);
    panel.append(scopes, delivery, list, composer);
    body.append(panel);
  }

  async function sendDirective(input, button) {
    const text = input.value.trim();
    if (!text || !activeId || directiveBusy) return;
    directiveBusy = true;
    button.disabled = true;
    try {
      const response = await fetch(`/api/theater/${activeId}/directives`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, scope: directiveScope, delivery: directiveDelivery }),
      });
      const next = await response.json();
      if (!response.ok) throw new Error(next.error || 'Could not send directive');
      input.value = '';
      renderState(next);
    } catch (error) {
      toast(error.message);
    } finally {
      directiveBusy = false;
      button.disabled = false;
    }
  }

  async function removeDirective(id) {
    try {
      const response = await fetch(`/api/theater/${activeId}/directives/${id}`, { method: 'DELETE' });
      const next = await response.json();
      if (!response.ok) throw new Error(next.error || 'Could not remove directive');
      renderState(next);
    } catch (error) {
      toast(error.message);
    }
  }

  async function renderSaved() {
    const body = $('sideBody');
    body.replaceChildren();
    try {
      const response = await fetch('/api/theater');
      const { sessions } = await response.json();
      if (!sessions?.length) {
        const empty = document.createElement('div');
        empty.className = 'empty-state';
        empty.textContent = 'No saved streams yet.';
        body.append(empty);
        return;
      }
      sessions.forEach((session) => {
        const card = document.createElement('button');
        card.type = 'button';
        card.className = 'scene-card';
        const icon = document.createElement('span');
        icon.className = 'scene-index';
        icon.textContent = '▶';
        const meta = document.createElement('span');
        meta.className = 'scene-meta';
        const title = document.createElement('strong');
        title.textContent = session.title || session.config?.prompt || 'Saved Stream';
        const detail = document.createElement('small');
        detail.textContent = `${(session.segments || []).length} scenes · ${session.status}`;
        meta.append(title, detail);
        const duration = document.createElement('span');
        duration.className = 'scene-duration';
        duration.textContent = formatTime(session.total_duration);
        card.append(icon, meta, duration);
        card.addEventListener('click', () => poll(session.id));
        body.append(card);
      });
    } catch {
      const empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.textContent = 'Could not load saved streams.';
      body.append(empty);
    }
  }

  function renderSide() {
    if (activeTab === 'scenes') renderScenes();
    else if (activeTab === 'direct') renderDirect();
    else renderSaved();
  }

  function renderState(next) {
    state = next;
    activeId = next.id;
    localStorage.setItem('wanTheaterSession', activeId);
    const running = activeStatuses.has(next.status);
    $('btnStop').disabled = !running;
    $('statusBadge').textContent = running ? 'LIVE' : String(next.status || 'idle').toUpperCase();
    if (currentIndex < 0) $('sceneTitle').textContent = next.title || next.config?.prompt || 'Endless Offline Theater';
    renderSide();
    if (currentIndex < 0 && next.segments?.length) {
      playScene(0);
    } else if (waitingForNext && next.segments?.[currentIndex + 1]) {
      playScene(currentIndex + 1);
    }
    const waitingWithoutMedia = !next.segments?.length || (waitingForNext && !next.segments?.[currentIndex + 1]);
    if (running && waitingWithoutMedia) interlude.show(next.attention || starterAttention());
    else if (!running) interlude.hide(captions && currentIndex >= 0);
    clearTimeout(pollTimer);
    if (running) pollTimer = setTimeout(() => poll(activeId), 1500);
  }

  async function poll(id) {
    if (!id) return;
    try {
      const response = await fetch(`/api/theater/${id}`);
      if (response.status === 404) {
        clearTimeout(pollTimer);
        if (localStorage.getItem('wanTheaterSession') === id) {
          localStorage.removeItem('wanTheaterSession');
        }
        if (activeId === id) {
          activeId = null;
          state = null;
          window.location.reload();
        } else {
          renderSide();
        }
        return;
      }
      if (!response.ok) throw new Error();
      renderState(await response.json());
    } catch {
      clearTimeout(pollTimer);
      pollTimer = setTimeout(() => poll(id), 2500);
    }
  }

  async function stopStory() {
    if (!activeId) return;
    $('btnStop').disabled = true;
    try {
      const response = await fetch(`/api/theater/${activeId}/stop`, { method: 'POST' });
      const next = await response.json();
      if (!response.ok) throw new Error(next.error || 'Could not stop stream');
      player.pause();
      renderState(next);
    } catch (error) {
      $('btnStop').disabled = false;
      toast(error.message);
    }
  }

  async function previewVoice() {
    const button = $('btnPreviewVoice');
    button.disabled = true;
    try {
      const response = await fetch('/api/theater/voice-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice: $('voiceSelect').value, language: $('languageSelect').value }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Voice preview failed');
      const audio = $('voicePreview');
      audio.src = `${mediaUrl(result.path)}&v=${Date.now()}`;
      audio.hidden = false;
      await audio.play();
    } catch (error) {
      toast(error.message);
    } finally {
      button.disabled = false;
    }
  }

  function updatePlayback() {
    const duration = player.duration || 0;
    const current = player.currentTime || 0;
    $('timeDisplay').textContent = `${formatTime(current)} / ${formatTime(duration)}`;
    $('scrubber').value = duration ? Math.round(current / duration * 1000) : 0;
    highlighter.update(current);
  }

  function bindEvents() {
    $('storyForm').addEventListener('submit', startStory);
    $('btnOpenSettings').addEventListener('click', () => $('settingsModal').classList.add('open'));
    $('btnCloseSettings').addEventListener('click', () => $('settingsModal').classList.remove('open'));
    $('settingsModal').addEventListener('click', (event) => {
      if (event.target === $('settingsModal')) $('settingsModal').classList.remove('open');
    });
    $('btnStop').addEventListener('click', stopStory);
    $('btnPreviewVoice').addEventListener('click', previewVoice);
    $('languageSelect').addEventListener('change', syncLanguageControls);
    $('translationSelect').addEventListener('change', syncLanguageControls);
    $('quickLanguageSelect').addEventListener('change', () => {
      $('languageSelect').value = $('quickLanguageSelect').value;
      syncLanguageControls();
    });
    $('quickTranslationSelect').addEventListener('change', () => {
      $('translationSelect').value = $('quickTranslationSelect').value;
      syncLanguageControls();
    });
    $('btnSwapLanguages').addEventListener('click', () => {
      const source = $('languageSelect').value;
      $('languageSelect').value = $('translationSelect').value;
      $('translationSelect').value = source;
      syncLanguageControls();
    });
    $('btnPlayPause').addEventListener('click', () => { if (player.paused) player.play().catch(() => {}); else player.pause(); });
    $('btnNextScene').addEventListener('click', () => playScene(currentIndex + 1));
    $('btnMute').addEventListener('click', () => { muted = !muted; player.muted = muted; $('btnMute').classList.toggle('active', muted); });
    $('btnCc').addEventListener('click', () => {
      captions = !captions;
      $('btnCc').classList.toggle('active', captions);
      syncCaptionVisibility();
    });
    $('btnFullscreen').addEventListener('click', () => {
      if (document.fullscreenElement) document.exitFullscreen?.();
      else $('playerShell').requestFullscreen?.();
    });
    $('scrubber').addEventListener('input', () => {
      if (player.duration) player.currentTime = Number($('scrubber').value) / 1000 * player.duration;
    });
    player.addEventListener('timeupdate', updatePlayback);
    player.addEventListener('loadedmetadata', updatePlayback);
    player.addEventListener('play', () => $('playerShell').classList.add('playing'));
    player.addEventListener('pause', () => $('playerShell').classList.remove('playing'));
    player.addEventListener('ended', advance);
    document.querySelectorAll('.side-tabs button').forEach((button) => {
      button.addEventListener('click', () => {
        activeTab = button.dataset.tab;
        document.querySelectorAll('.side-tabs button').forEach((item) => item.classList.toggle('active', item === button));
        renderSide();
      });
    });
    document.addEventListener('keydown', (event) => {
      if (/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || '')) return;
      if (event.key.toLowerCase() === 'c') $('btnCc').click();
      if (event.key.toLowerCase() === 'f') $('btnFullscreen').click();
    });
  }

  document.addEventListener('DOMContentLoaded', async () => {
    bindEvents();
    try {
      await loadOptions();
      const remembered = localStorage.getItem('wanTheaterSession');
      if (!remembered) {
        renderSide();
        return;
      }
      const response = await fetch('/api/theater');
      if (!response.ok) throw new Error('Could not restore saved streams');
      const payload = await response.json();
      const session = (payload.sessions || []).find((item) => item.id === remembered);
      if (session) {
        renderState(session);
      } else {
        localStorage.removeItem('wanTheaterSession');
        renderSide();
      }
    } catch (error) {
      toast(error.message);
    }
  });
}());
