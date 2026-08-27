(function () {
  'use strict';

  function normalize(value) {
    return String(value || '').toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
  }

  function words(value) {
    return (String(value || '').match(/[\p{L}\p{N}]+/gu) || []).map((word) => word.toLowerCase());
  }

  function substitutionCost(left, right) {
    if (left === right) return 0;
    if (left.length >= 3 && right.length >= 3 && (left.startsWith(right) || right.startsWith(left))) return 0.35;
    return 1;
  }

  function alignTimings(text, timings) {
    const source = words(text);
    const recognized = timings.map((timing) => normalize(timing.value));
    if (!source.length) return timings.map(() => -1);
    const gap = 0.7;
    const costs = Array.from({ length: source.length + 1 }, () => new Float64Array(recognized.length + 1));
    const moves = Array.from({ length: source.length + 1 }, () => new Uint8Array(recognized.length + 1));
    for (let s = 1; s <= source.length; s += 1) { costs[s][0] = s * gap; moves[s][0] = 1; }
    for (let t = 1; t <= recognized.length; t += 1) { costs[0][t] = t * gap; moves[0][t] = 2; }
    for (let s = 1; s <= source.length; s += 1) {
      for (let t = 1; t <= recognized.length; t += 1) {
        const diagonal = costs[s - 1][t - 1] + substitutionCost(source[s - 1], recognized[t - 1]);
        const sourceOnly = costs[s - 1][t] + gap;
        const timingOnly = costs[s][t - 1] + gap;
        if (diagonal <= sourceOnly && diagonal <= timingOnly) {
          costs[s][t] = diagonal; moves[s][t] = 0;
        } else if (sourceOnly <= timingOnly) {
          costs[s][t] = sourceOnly; moves[s][t] = 1;
        } else {
          costs[s][t] = timingOnly; moves[s][t] = 2;
        }
      }
    }
    const indices = Array(recognized.length).fill(-1);
    let s = source.length;
    let t = recognized.length;
    while (s > 0 || t > 0) {
      const move = moves[s][t];
      if (s > 0 && t > 0 && move === 0) { indices[t - 1] = s - 1; s -= 1; t -= 1; }
      else if (s > 0 && (t === 0 || move === 1)) s -= 1;
      else t -= 1;
    }
    for (let index = 0; index < indices.length; index += 1) {
      if (indices[index] >= 0) continue;
      let previous = index - 1;
      let next = index + 1;
      while (previous >= 0 && indices[previous] < 0) previous -= 1;
      while (next < indices.length && indices[next] < 0) next += 1;
      if (previous >= 0 && next < indices.length) {
        const fraction = (index - previous) / (next - previous);
        indices[index] = Math.round(indices[previous] + fraction * (indices[next] - indices[previous]));
      } else if (previous >= 0) indices[index] = indices[previous];
      else if (next < indices.length) indices[index] = indices[next];
      else indices[index] = 0;
    }
    for (let index = 0; index < indices.length; index += 1) {
      indices[index] = Math.min(source.length - 1, Math.max(index ? indices[index - 1] : 0, indices[index]));
    }
    return indices;
  }

  function estimatedTimings(text, duration) {
    const tokens = String(text || '').match(/[\p{L}\p{N}]+/gu) || [];
    const weights = tokens.map((token) => Math.max(1, token.length));
    const total = weights.reduce((sum, value) => sum + value, 0) || 1;
    let cursor = 0;
    return tokens.map((value, index) => {
      const start = duration * cursor / total;
      cursor += weights[index];
      return { value, start, end: duration * cursor / total };
    });
  }

  function timingIndexAt(timings, seconds) {
    if (!timings.length || seconds < 0) return -1;
    if (seconds < timings[0].start) return 0;
    if (seconds >= timings[timings.length - 1].start) return timings.length - 1;
    for (let index = 0; index < timings.length; index += 1) {
      if (seconds >= timings[index].start && seconds < timings[index].end) return index;
      if (index + 1 < timings.length && seconds < timings[index + 1].start) return index;
    }
    return timings.length - 1;
  }

  class CaptionHighlighter {
    constructor(container, seek) {
      this.container = container;
      this.seek = seek;
      this.nodes = [];
      this.sentences = [];
      this.timings = [];
      this.mapping = [];
      this.sourceTiming = [];
      this.activeIndex = -1;
      this.activeSentence = -1;
      this.text = '';
    }

    render(scene, duration) {
      this.container.replaceChildren();
      this.nodes = [];
      this.sentences = [];
      this.activeIndex = -1;
      this.activeSentence = -1;
      const pairs = Array.isArray(scene?.narration_sentences) && scene.narration_sentences.length
        ? scene.narration_sentences
        : [{ original: scene?.narration || '' }];
      let wordIndex = 0;
      const orderedText = [];
      const addLine = (parent, text, className, sentenceIndex) => {
        const line = document.createElement('div');
        line.className = className;
        const tokens = String(text || '').split(/([^\p{L}\p{N}]+)/gu);
        tokens.forEach((token) => {
          if (!token) return;
          if (!/[\p{L}\p{N}]/u.test(token)) { line.append(document.createTextNode(token)); return; }
          const span = document.createElement('span');
          span.className = 'speech-word pending';
          span.textContent = token;
          span.dataset.wordIndex = String(wordIndex);
          span.dataset.sentenceIndex = String(sentenceIndex);
          span.addEventListener('click', () => this.seek(this.sourceTiming[Number(span.dataset.wordIndex)] || 0));
          this.nodes.push(span);
          wordIndex += 1;
          line.append(span);
        });
        orderedText.push(String(text || '').trim());
        parent.append(line);
      };
      pairs.forEach((pair, sentenceIndex) => {
        const block = document.createElement('div');
        block.className = 'caption-pair';
        block.dataset.sentenceIndex = String(sentenceIndex);
        addLine(block, pair.original, 'caption-original', sentenceIndex);
        if (pair.translation) addLine(block, pair.translation, 'caption-translation', sentenceIndex);
        this.sentences.push(block);
        this.container.append(block);
      });
      this.text = scene?.spoken_text || orderedText.filter(Boolean).join(' ');
      const exact = Array.isArray(scene?.word_timestamps) ? scene.word_timestamps : [];
      this.timings = exact.length ? exact : estimatedTimings(this.text, duration || Math.max(1, this.text.length / 15));
      this.mapping = exact.length ? alignTimings(this.text, this.timings) : this.timings.map((_, index) => index);
      this.sourceTiming = Array(this.nodes.length).fill(null);
      this.mapping.forEach((sourceIndex, timingIndex) => {
        if (sourceIndex >= 0 && this.sourceTiming[sourceIndex] === null) this.sourceTiming[sourceIndex] = Number(this.timings[timingIndex]?.start || 0);
      });
      for (let index = 1; index < this.sourceTiming.length; index += 1) {
        if (this.sourceTiming[index] === null) this.sourceTiming[index] = this.sourceTiming[index - 1] ?? 0;
      }
      if (this.sourceTiming.length && this.sourceTiming[0] === null) this.sourceTiming[0] = 0;
    }

    update(seconds) {
      const timingIndex = timingIndexAt(this.timings, seconds);
      const next = timingIndex >= 0 ? Math.min(this.nodes.length - 1, Math.max(0, this.mapping[timingIndex] ?? timingIndex)) : -1;
      if (next === this.activeIndex) return;
      this.nodes.forEach((node, index) => {
        node.classList.toggle('active', index === next);
        node.classList.toggle('spoken', index < next);
        node.classList.toggle('pending', index > next);
      });
      this.activeIndex = next;
      const sentence = next >= 0 ? Number(this.nodes[next]?.dataset.sentenceIndex) : -1;
      if (sentence !== this.activeSentence) {
        this.sentences.forEach((node, index) => node.classList.toggle('active-sentence', index === sentence));
        this.sentences[sentence]?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        this.activeSentence = sentence;
      }
    }
  }

  window.CaptionHighlighter = CaptionHighlighter;
  window.TheaterHighlight = { alignTimings, estimatedTimings, timingIndexAt };
}());
