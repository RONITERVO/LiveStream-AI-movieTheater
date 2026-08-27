(function () {
  'use strict';

  function formatDuration(seconds) {
    const value = Math.max(0, Math.round(Number(seconds) || 0));
    if (!value) return '';
    const minutes = Math.floor(value / 60);
    return minutes ? `${minutes}:${String(value % 60).padStart(2, '0')}` : `${value}s`;
  }

  function validPairs(value) {
    if (!Array.isArray(value)) return [];
    return value.filter((pair) => pair && String(pair.source || '').trim() && String(pair.translation || '').trim())
      .slice(0, 16)
      .map((pair) => ({ source: String(pair.source).trim(), translation: String(pair.translation).trim() }));
  }

  function pairDwell(pair) {
    const characters = Array.from(`${pair?.source || ''}${pair?.translation || ''}`).length;
    return Math.min(5200, Math.max(2600, 1900 + characters * 45));
  }

  class LearningInterlude {
    constructor(elements) {
      this.shell = elements.shell;
      this.window = elements.window;
      this.transcript = elements.transcript;
      this.root = elements.root;
      this.pair = elements.pair;
      this.source = elements.source;
      this.translation = elements.translation;
      this.meta = elements.meta;
      this.pairs = [];
      this.index = 0;
      this.key = '';
      this.timer = null;
      this.active = false;
    }

    show(attention) {
      const pairs = validPairs(attention?.pairs);
      if (!pairs.length) return;
      const key = JSON.stringify(pairs);
      const changed = key !== this.key;
      if (changed) {
        this.key = key;
        this.pairs = pairs;
        this.index = 0;
      }
      this.active = true;
      this.shell.classList.add('learning');
      this.window.classList.add('learning');
      this.window.hidden = false;
      this.transcript.hidden = true;
      this.root.hidden = false;
      const eta = formatDuration(attention?.eta_seconds);
      this.meta.textContent = [eta ? `~${eta}` : '', String(attention?.detail || '')].filter(Boolean).join(' · ');
      this.render();
      if (changed) {
        clearTimeout(this.timer);
        this.timer = null;
      }
      if (!this.timer) this.schedule();
    }

    hide(showTranscript) {
      this.active = false;
      this.shell.classList.remove('learning');
      this.window.classList.remove('learning');
      this.root.hidden = true;
      this.transcript.hidden = false;
      this.window.hidden = !showTranscript;
      clearTimeout(this.timer);
      this.timer = null;
    }

    schedule() {
      if (!this.active || !this.pairs.length) return;
      this.timer = setTimeout(() => {
        this.timer = null;
        this.next();
        this.schedule();
      }, pairDwell(this.pairs[this.index]));
    }

    next() {
      if (!this.active || !this.pairs.length) return;
      this.index = (this.index + 1) % this.pairs.length;
      this.pair.classList.remove('arrive');
      requestAnimationFrame(() => {
        this.render();
        this.pair.classList.add('arrive');
      });
    }

    render() {
      const pair = this.pairs[this.index] || this.pairs[0];
      if (!pair) return;
      this.source.textContent = pair.source;
      this.translation.textContent = pair.translation;
    }
  }

  window.LearningInterlude = LearningInterlude;
  window.TheaterInterlude = { formatDuration, pairDwell, validPairs };
}());
