'use strict';
const $ = id => document.getElementById(id);
let nextQuestionId = 0, questionRows = [], images = [], history = [], activeRun = null;
let apiModel = null, codeModel = null, poolingDefaultApplied = false;
let running = false, imageLoading = false, elapsedFrame = 0, runSequence = 0, pastedImageSequence = 0;
let codeFormat = 'curl', autoRunTimer = 0, autoRunPending = false;
const editedQuestions = new Set();
const typeNames = {noul: 'Yes / no', choice: 'Choose one', score: 'Score'};
const imageScalings = {
  compact: {name: 'Compact', width: 256, height: 256},
  mcga: {name: 'MCGA', width: 320, height: 200},
  sd: {name: '480p', width: 640, height: 480},
  hd: {name: 'HD · 720p', width: 1280, height: 720},
  fullhd: {name: 'Full HD · 1080p', width: 1920, height: 1080},
  qhd: {name: 'QHD · 1440p', width: 2560, height: 1440},
  uhd: {name: '4K UHD', width: 3840, height: 2160},
  original: {name: 'Original', original: true}
};
const isMac = /Mac|iPhone|iPad/.test(navigator.platform);
const storage = {
  get(key) {try {return localStorage.getItem(key);} catch {return null;}},
  set(key, value) {try {localStorage.setItem(key, value);} catch {}}
};

function updateScalingNote() {
  const preset = imageScalings[$('image-scaling').value];
  $('image-note').textContent = preset.original
    ? 'Upload the original file without resizing or changing its format. Large images use more input tokens.'
    : `Fit within ${preset.width} × ${preset.height}, rotating the frame for portrait photos. Images are downscaled in this browser without enlargement, keeping their original file format.`;
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function fit(textarea) {textarea.style.height = 'auto'; textarea.style.height = `${textarea.scrollHeight + 2}px`;}
function showError(message, field) {
  $('form-error').textContent = message; $('form-error').hidden = false;
  if (field) {field.setAttribute('aria-invalid', 'true'); field.focus();}
}
function clearError() {
  $('form-error').hidden = true;
  document.querySelectorAll('[aria-invalid]').forEach(el => el.removeAttribute('aria-invalid'));
}

/* Editing state: answers dim when their question changes after a run. */
function markEdited(rowId) {
  if (rowId) editedQuestions.add(rowId); else questionRows.forEach(row => editedQuestions.add(row.id));
  if (activeRun && !running) {
    document.querySelectorAll('.answer-card').forEach(card => {
      if (!rowId || card.dataset.link === rowId) card.classList.add('stale');
    });
    if (!activeRun.error) {$('run-state').textContent = `Edited since run ${activeRun.number}`; $('run-state').className = 'run-state stale';}
  }
  refreshCode(); scheduleAutoRun();
}
function scheduleAutoRun() {
  if (!$('auto-run').checked) return;
  clearTimeout(autoRunTimer);
  autoRunTimer = setTimeout(() => {
    if (running || imageLoading) {autoRunPending = true; return;}
    if (collectQuestions(false)) $('decision-form').requestSubmit();
  }, 1000);
}

/* Questions */
function addQuestion(initial = {}, focus = false) {
  if (questionRows.length >= 64) {showError('A request can contain up to 64 questions.'); return;}
  const id = `q${++nextQuestionId}`;
  const card = node('div', 'question-card'); card.dataset.link = id;
  const toolbar = node('div', 'question-toolbar');
  const number = node('span', 'question-number');
  const seg = node('div', 'seg'); seg.setAttribute('role', 'group');
  const typeButtons = Object.entries(typeNames).map(([value, name]) => {
    const button = node('button', '', name); button.type = 'button'; button.dataset.type = value; seg.append(button); return button;
  });
  const remove = node('button', 'remove-question', '×'); remove.type = 'button';
  toolbar.append(number, seg, remove);
  const label = node('label', 'sr-only'); label.htmlFor = `${id}-text`;
  const text = document.createElement('textarea'); text.id = `${id}-text`; text.rows = 1; text.className = 'auto';
  text.placeholder = 'e.g., Is there a dog in the image?'; text.value = initial.instructions || '';
  text.setAttribute('aria-describedby', 'form-error');
  const optionsField = node('div', 'options-field');
  const chipList = node('div', 'chip-list');
  const chipInput = document.createElement('input'); chipInput.className = 'chip-input'; chipInput.id = `${id}-options`;
  chipInput.setAttribute('aria-describedby', `${id}-options-help form-error`);
  const help = node('p', 'help'); help.id = `${id}-options-help`;
  optionsField.append(chipList, help);
  card.append(toolbar, label, text, optionsField);
  const row = {id, card, text, type: 'noul', options: [], typeButtons, chipList, chipInput, optionsField, help, number, remove, label, seg};

  for (const button of typeButtons) button.addEventListener('click', () => {setType(row, button.dataset.type); markEdited(id);});
  text.addEventListener('input', () => {fit(text); markEdited(id);});
  chipInput.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.isComposing) {
      event.preventDefault(); if (event.metaKey || event.ctrlKey) return;
      if (addOptions(row, [chipInput.value])) chipInput.value = '';
    } else if (event.key === 'Backspace' && !chipInput.value && row.options.length) {
      row.options.pop(); renderChips(row); markEdited(id);
    }
  });
  chipInput.addEventListener('paste', event => {
    const pasted = event.clipboardData?.getData('text') || '';
    if (!pasted.includes('\n')) return;
    event.preventDefault(); addOptions(row, pasted.split('\n'));
  });
  chipInput.addEventListener('blur', () => {if (chipInput.value.trim() && addOptions(row, [chipInput.value], true)) chipInput.value = '';});
  remove.addEventListener('click', () => {
    if (questionRows.length === 1) return;
    const index = questionRows.indexOf(row); questionRows.splice(index, 1); card.remove();
    updateQuestions(); refreshCode(); questionRows[Math.min(index, questionRows.length - 1)].text.focus();
  });

  questionRows.push(row); $('questions').append(card);
  setType(row, initial.type || 'noul');
  if (initial.options) {row.options = [...initial.options]; renderChips(row);}
  updateQuestions(); fit(text);
  if (focus) text.focus();
}
function setType(row, type) {
  row.type = type;
  row.typeButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.type === type)));
  row.optionsField.hidden = type === 'noul';
  row.chipInput.placeholder = type === 'score' ? 'Add level ↵' : 'Add answer ↵';
  row.chipInput.setAttribute('aria-label', type === 'score' ? 'Add score level' : 'Add answer option');
  row.help.textContent = type === 'score'
    ? 'Score levels, lowest first. Press Enter to add one, or paste a list.'
    : 'Press Enter to add an answer, or paste a list with one per line.';
  renderChips(row);
}
function addOptions(row, values, quiet = false) {
  clearError(); let added = false;
  for (const raw of values) {
    const value = raw.trim(); if (!value) continue;
    if (row.options.includes(value)) {if (!quiet) showError(`“${value}” is already an option. Give each answer a different name.`); continue;}
    row.options.push(value); added = true;
  }
  if (added) {renderChips(row); markEdited(row.id); row.chipInput.focus();}
  return added;
}
function renderChips(row) {
  row.chipList.replaceChildren();
  row.options.forEach((option, index) => {
    const chip = node('span', 'chip');
    if (row.type === 'score') chip.append(node('span', 'level', String(index)));
    chip.append(document.createTextNode(option));
    const remove = node('button', '', '×'); remove.type = 'button';
    remove.setAttribute('aria-label', `Remove ${option}`);
    remove.addEventListener('click', () => {row.options.splice(index, 1); renderChips(row); markEdited(row.id); row.chipInput.focus();});
    chip.append(remove); row.chipList.append(chip);
  });
  row.chipList.append(row.chipInput);
}
function updateQuestions() {
  questionRows.forEach((row, index) => {
    row.number.textContent = index + 1;
    row.label.textContent = `Question ${index + 1}`;
    row.seg.setAttribute('aria-label', `Answer type for question ${index + 1}`);
    row.remove.setAttribute('aria-label', `Remove question ${index + 1}`);
    row.remove.hidden = questionRows.length === 1;
  });
  $('question-count').textContent = `${questionRows.length} question${questionRows.length === 1 ? '' : 's'}`;
}
// Returns the API questions object, or null when the editor is incomplete.
function collectQuestions(report) {
  const questions = {};
  for (const row of questionRows) {
    const instructions = row.text.value.trim();
    if (!instructions) {if (report) showError('Enter a question before asking Clef.', row.text); return null;}
    const question = {type: row.type, instructions};
    if (row.type !== 'noul') {
      if (row.options.length < 2) {if (report) showError('Add at least two answer options. Press Enter after each one.', row.chipInput); return null;}
      question.criteria = row.type === 'score' ? [...row.options] : Object.fromEntries(row.options.map(value => [value, value]));
    }
    questions[row.id] = question;
  }
  return questions;
}
function requestState() {return $('context').value.trim() || (images.length ? 'Inspect all supplied images.' : 'Answer the supplied questions.');}

/* Images */
function renderImages() {
  $('image-previews').replaceChildren();
  images.forEach((item, index) => {
    const preview = node('div', 'image-preview'); preview.title = item.name;
    const img = document.createElement('img'); img.src = item.data; img.alt = `Selected image: ${item.name}`;
    const remove = node('button', 'image-remove', '×'); remove.type = 'button';
    remove.setAttribute('aria-label', `Remove image ${item.name}`);
    remove.addEventListener('click', () => {images.splice(index, 1); renderImages(); clearError(); markEdited(); $('image-status').textContent = `${images.length} image${images.length === 1 ? '' : 's'} attached.`; $('image-input').value = ''; $('image-input').focus();});
    preview.append(img, remove); $('image-previews').append(preview);
  });
}
async function acceptImages(files) {
  if (running || imageLoading) return;
  clearError(); const incoming = [...files];
  // Allow selecting the same file again after a rejected upload.
  $('image-input').value = '';
  if (incoming.length + images.length > 16) {showError('Choose up to 16 images. Remove an image before adding another.'); return;}
  for (const file of incoming) {
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {showError(`${file.name}: choose a PNG, JPEG, or WebP image.`); return;}
    if (file.size > 10 * 1024 * 1024) {showError(`${file.name} exceeds the 10 MB image limit.`); return;}
  }
  imageLoading = true; $('editor-fields').disabled = true; $('example').disabled = true;
  $('run-button').disabled = true; $('run-button-text').textContent = 'Reading images…';
  try {
    const loaded = await Promise.all(incoming.map(file => new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const image = new Image();
        image.onload = () => {
          if (image.naturalWidth * image.naturalHeight > 20000000) reject(new Error(`${file.name} exceeds 20 million pixels.`));
          else resolve({name: file.name, data: reader.result, image, format: file.type, bytes: file.size, prepared: null});
        };
        image.onerror = () => reject(new Error(`${file.name} could not be read as an image.`));
        image.src = reader.result;
      };
      reader.onerror = () => reject(new Error(`Could not read ${file.name}.`)); reader.readAsDataURL(file);
    })));
    images.push(...loaded); renderImages(); markEdited();
    $('image-status').textContent = `${loaded.length} image${loaded.length === 1 ? '' : 's'} added. ${images.length} attached.`;
  } catch (error) {showError(error.message);}
  finally {
    imageLoading = false; $('editor-fields').disabled = false; $('example').disabled = false;
    $('run-button').disabled = false; $('run-button-text').textContent = 'Ask Clef'; $('image-input').value = '';
  }
}

/* Server */
async function checkConnection() {
  if (running) return;
  $('connection-text').textContent = 'Connecting'; $('connection').className = 'connection';
  try {
    const response = await fetch('/health', {signal: AbortSignal.timeout(5000), cache: 'no-store'});
    if (!response.ok) throw new Error('unavailable'); const health = await response.json();
    $('connection').classList.add('online'); $('connection-text').textContent = 'Server ready';
    $('connection').title = `${health.model} at ${location.host}. Click to check again.`;
    const imageLimit = health.max_input_tokens_with_images ?? health.max_input_tokens;
    const estimateLabel = health.context_limit?.estimated ? 'Estimated limit: ' : 'Limit: ';
    $('limit-note').textContent = estimateLabel + (imageLimit < health.max_input_tokens
      ? `${health.max_input_tokens.toLocaleString()} text input tokens; ${imageLimit.toLocaleString()} with images`
      : `${health.max_input_tokens.toLocaleString()} total input tokens`);
    apiModel = health.model;
    if (!poolingDefaultApplied) {
      $('image-pooling').checked = !!health.image_input.pooling_default;
      poolingDefaultApplied = true;
    }
    $('model-name').textContent = health.model === 'Cloudflare/clef' ? 'Clef 27B' : 'Clef Flash';
    $('hardware-note').textContent = `${health.gpu.replace('NVIDIA GeForce ', '')} / ${health.quantization.toUpperCase()}`;
    if (!codeModel) void loadCodeModel();
    refreshCode();
  } catch {
    $('connection').classList.add('offline'); $('connection-text').textContent = 'Server unavailable';
    $('connection').title = 'The server could not be reached. Click to retry.';
  }
}
// Code examples use the short model id that /v1/models advertises; the server accepts either form.
async function loadCodeModel() {
  try {
    const response = await fetch('/v1/models', {signal: AbortSignal.timeout(5000), cache: 'no-store'});
    if (response.ok) {codeModel = (await response.json()).data?.[0]?.id || null; refreshCode();}
  } catch {}
}

/* Results */
function percent(value) {return `${(value * 100).toFixed(1)}%`;}
function duration(ms) {return `${(ms / 1000).toFixed(3)} s`;}
function showProbabilities(container, entries, winner) {
  const probabilities = node('div', 'probabilities');
  for (const [label, probability] of entries) {
    const row = node('div', `prob-row${label === winner ? ' winner' : ''}`);
    const name = node('span', 'prob-label', label);
    const track = node('div', 'prob-track'); track.setAttribute('aria-hidden', 'true');
    const fill = node('div', 'prob-fill'); fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`; track.append(fill);
    row.append(name, track, node('span', 'prob-value', percent(probability))); probabilities.append(row);
  }
  container.append(probabilities);
}
function usageCacheNote(usage) {
  const images = usage?.image_feature_cache_hits || 0;
  const processed = usage?.image_preprocess_cache_hits || 0;
  const note = `${usage?.queue_wait_ms != null ? ` Queue wait: ${duration(usage.queue_wait_ms)}.` : ''}${images ? ` Reused ${images} image feature(s).` : ''}${processed ? ` Reused preprocessing for ${processed} image(s).` : ''}`;
  return note + (usage?.prefix_cache === 'hit' ? ` Reused ${usage.reused_prefix_tokens ?? usage.prefix_tokens} context tokens${usage.vision_prefix_reused ? ' and image encoding' : ''}.` : '');
}
// Reduce any answer type to its displayed winner, confidence and probability bars.
function summarize(answer) {
  if (answer.type === 'noul') {
    const yes = answer.noul, winner = yes >= .5 ? 'Yes' : 'No';
    return {winner, confidence: Math.max(yes, 1 - yes), entries: [['Yes', yes], ['No', 1 - yes]], subtitle: `Probability of yes ${percent(yes)}`};
  }
  if (answer.type === 'choice') {
    return {winner: answer.choice, confidence: answer.probabilities[answer.choice],
      entries: Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1]), subtitle: `Confidence ${percent(answer.probabilities[answer.choice])}`};
  }
  const entries = Object.entries(answer.probabilities).map(([index, value]) => [answer.legend[index], value]);
  const [winner, confidence] = entries.reduce((best, current) => current[1] > best[1] ? current : best);
  return {winner, confidence, entries, subtitle: `Expected score ${answer.score.toFixed(2)} on a 0–${entries.length - 1} scale`};
}
// Compare against the most recent earlier successful run that asked the same question.
function delta(run, id) {
  const current = run.response.answers[id], question = run.questions[id];
  const earlier = history.slice(history.indexOf(run) + 1).find(item => !item.error && item.response.answers[id]
    && item.questions[id]?.type === question.type && JSON.stringify(item.questions[id].criteria) === JSON.stringify(question.criteria));
  if (!earlier) return null;
  const now = summarize(current), before = summarize(earlier.response.answers[id]);
  if (now.winner !== before.winner) return {className: 'down', text: `was ${before.winner}`, title: `Run ${earlier.number} answered ${before.winner}`};
  // Round first so a tiny drop reads as "0.0 pts" rather than "−0.0 pts".
  const points = Math.round((now.confidence - before.confidence) * 1000) / 10;
  return {className: Math.abs(points) < .5 ? 'flat' : points > 0 ? 'up' : 'down',
    text: points === 0 ? '0.0 pts' : `${points > 0 ? '+' : '−'}${Math.abs(points).toFixed(1)} pts`, title: `Change in confidence since run ${earlier.number}`};
}
function renderRun(run) {
  activeRun = run; editedQuestions.clear();
  $('elapsed').textContent = (run.totalMs / 1000).toFixed(3);
  $('run-label').textContent = `Run ${run.number}${run.imageNames.length ? ' · ' + imageScalings[run.imageScaling].name + (run.imagePooling ? ' · 2×2 pooling' : '') : ''}${run.response?.usage?.prefix_cache === 'hit' ? ' · prefix cache hit' : ''}`;
  $('run-state').textContent = run.error ? 'Failed' : `Run ${run.number}`; $('run-state').className = `run-state ${run.error ? 'failed' : 'success'}`;
  $('timer-description').textContent = run.error ? '' : `Client image preparation: ${duration(run.clientImageMs || 0)}. Upload: ${((run.uploadBytes || 0) / 1024).toFixed(0)} KiB.${usageCacheNote(run.response?.usage)}`;
  const usage = run.response?.usage || {};
  $('server-time').textContent = typeof usage.latency_ms === 'number' ? duration(usage.latency_ms) : '—';
  $('input-tokens').textContent = typeof usage.input_tokens === 'number' ? usage.input_tokens.toLocaleString() : '—';
  $('gpu-memory').textContent = typeof usage.peak_allocated_mib === 'number' ? `${(usage.peak_allocated_mib / 1024).toFixed(2)} GiB` : '—';
  $('results').replaceChildren();
  if (run.error) {
    const box = node('div', 'run-failure'); box.append(node('h3', '', 'Request failed'), node('p', '', run.error)); $('results').append(box);
  } else {
    const ids = Object.keys(run.questions);
    for (const [id, answer] of Object.entries(run.response.answers)) {
      const question = run.questions[id], summary = summarize(answer);
      const card = node('article', 'answer-card'); card.dataset.link = id;
      const heading = node('div', 'answer-heading');
      heading.append(node('span', 'answer-number', String(ids.indexOf(id) + 1)), node('span', '', question?.instructions || id), node('span', 'answer-type', typeNames[answer.type] || answer.type));
      const line = node('div', 'answer-line');
      line.append(node('span', 'answer-value', summary.winner), node('span', 'answer-subtitle', summary.subtitle));
      const change = delta(run, id);
      if (change) {const badge = node('span', `delta ${change.className}`, change.text); badge.title = change.title; line.append(badge);}
      card.append(heading, line);
      showProbabilities(card, summary.entries, summary.winner);
      card.append(feedbackControls(run, id));
      $('results').append(card);
    }
  }
  const linked = document.querySelector('.question-card.linked');
  if (linked) highlight(linked.dataset.link, false);
  renderHistory(); refreshCode();
}
function feedbackControls(run, id) {
  const wrap = node('div', 'feedback'); wrap.append(node('span', '', 'Is this right?'));
  for (const [value, label] of [['good', 'Correct'], ['bad', 'Wrong']]) {
    const button = node('button', value, label); button.type = 'button';
    button.setAttribute('aria-pressed', String(run.feedback[id] === value));
    button.addEventListener('click', () => {
      run.feedback[id] = run.feedback[id] === value ? undefined : value;
      wrap.querySelectorAll('button').forEach(item => item.setAttribute('aria-pressed', String(run.feedback[id] === item.className)));
      if (codeFormat === 'raw') refreshCode();
    });
    wrap.append(button);
  }
  return wrap;
}
function renderHistory() {
  const spark = $('spark'); spark.replaceChildren();
  spark.hidden = history.length === 0; $('clear-history').hidden = history.length === 0;
  const max = Math.max(...history.map(run => run.totalMs), 1);
  for (const run of [...history].reverse()) {
    const bar = node('button', `${activeRun === run ? 'selected' : ''}${run.error ? ' failed' : ''}`); bar.type = 'button';
    bar.style.height = `${Math.max(10, run.totalMs / max * 100)}%`;
    const label = `Run ${run.number}: ${run.error ? 'failed' : duration(run.totalMs)}${run.error ? '' : `, server ${duration(run.response.usage.latency_ms)}`}`;
    bar.title = label; bar.setAttribute('aria-label', `Show ${label}`);
    bar.disabled = running;
    bar.addEventListener('click', () => {if (!running) {renderRun(run); $('result-announcement').textContent = `Showing run ${run.number}.`;}});
    spark.append(bar);
  }
}
// Link a question to its answer so you can see what you're changing.
function highlight(id, scroll) {
  document.querySelectorAll('[data-link]').forEach(element => element.classList.toggle('linked', element.dataset.link === id));
  if (scroll) document.querySelector(`.answer-card[data-link="${id}"]`)?.scrollIntoView({block: 'nearest', behavior: 'smooth'});
}

/* Code examples built from the current editor and the runtime URL */
const pythonCode = Symbol('python code');
function apiBase() {return location.origin;}
function exampleRequest(forPython) {
  const questions = {};
  for (const row of questionRows) {
    const question = {type: row.type, instructions: row.text.value.trim()};
    if (row.type === 'choice') question.criteria = Object.fromEntries(row.options.map(value => [value, value]));
    if (row.type === 'score') question.criteria = [...row.options];
    questions[row.id] = question;
  }
  const request = {model: codeModel || apiModel || 'clef-flash'};
  const context = $('shared-context').value.trim();
  if (context) request.context = context;
  request.state = requestState();
  if (forPython && images.length) {
    request.images = {[pythonCode]: '[image_data_url(path) for path in IMAGE_FILES]'};
    request.media_kwargs = {images_kwargs: {do_resize: true, min_pixels: 1024, max_pixels: 20000000}};
  }
  request.image_pooling = $('image-pooling').checked;
  request.questions = questions;
  return request;
}
function pythonLiteral(value, indent = '') {
  const inner = `${indent}    `;
  if (value === null) return 'None';
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (typeof value !== 'object') return JSON.stringify(value);
  if (value[pythonCode]) return value[pythonCode];
  if (Array.isArray(value)) return value.length ? `[\n${value.map(item => inner + pythonLiteral(item, inner)).join(',\n')},\n${indent}]` : '[]';
  const entries = Object.entries(value);
  return entries.length ? `{\n${entries.map(([key, item]) => `${inner}${JSON.stringify(key)}: ${pythonLiteral(item, inner)}`).join(',\n')},\n${indent}}` : '{}';
}
function curlExample() {
  return `curl ${apiBase()}/v1/systemone \\
  -H 'Content-Type: application/json' \\
  --data-binary @- <<'JSON'
${JSON.stringify(exampleRequest(false), null, 2)}
JSON`;
}
function pythonExample() {
  const imageSetup = images.length ? `IMAGE_FILES = [${images.map(item => JSON.stringify(item.name)).join(', ')}]


def image_data_url(path):
    # The server accepts PNG, JPEG or WebP images as base64 data URLs.
    mime = mimetypes.guess_type(path)[0]
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()

` : '';
  return `${images.length ? 'import base64\n' : ''}import json
${images.length ? 'import mimetypes\n' : ''}import urllib.request
${images.length ? 'from pathlib import Path\n' : ''}
URL = "${apiBase()}/v1/systemone"
${imageSetup}
request = ${pythonLiteral(exampleRequest(true))}

http_request = urllib.request.Request(
    URL,
    data=json.dumps(request).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(http_request, timeout=600) as response:
    result = json.load(response)

for question_id, answer in result["answers"].items():
    if answer["type"] == "noul":
        print(f"{question_id}: probability of yes {answer['noul']:.1%}")
    elif answer["type"] == "choice":
        print(f"{question_id}: {answer['choice']}")
    else:
        print(f"{question_id}: score {answer['score']:.2f}")

usage = result["usage"]
print(f"Server time {usage['latency_ms'] / 1000:.3f} s, {usage['input_tokens']} input tokens")`;
}
function rawExample() {
  if (!activeRun) return '// Run a request to see the raw response here.';
  if (activeRun.error) return `// Run ${activeRun.number} failed: ${activeRun.error}`;
  return JSON.stringify(activeRun.response, null, 2);
}
const codeExamples = {curl: curlExample, python: pythonExample, raw: rawExample};
function codeNote() {
  const count = images.length, plural = count === 1 ? '' : 's';
  if (codeFormat === 'curl') return count ? `This curl request leaves out the ${count} attached image${plural}, which must be sent as base64 data URLs. The Python example reads them from files.` : `Sends the request in this editor to ${apiBase()}.`;
  if (codeFormat === 'python') return `Uses only the Python standard library.${count ? ` It sends the original image file${plural}; this page resizes images to ${imageScalings[$('image-scaling').value].name} first, so input token counts can differ.` : ''}`;
  return activeRun ? `Response from run ${activeRun.number}.` : '';
}
function highlightCode(code) {
  const pattern = /(https?:\/\/\S+)|(^\/\/[^\n]*|#[^\n]*)|("(?:[^"\\\n]|\\.)*"|'[^'\n]*')(\s*:)?|\b(\d+(?:\.\d+)?)\b|\b(import|from|def|return|for|in|with|as|if|elif|else|True|False|None|true|false|null)\b/gm;
  const fragment = document.createDocumentFragment();
  let last = 0, match;
  const span = (className, text) => {const element = node('span', className, text); fragment.append(element);};
  while ((match = pattern.exec(code))) {
    fragment.append(code.slice(last, match.index)); last = pattern.lastIndex;
    if (match[1]) fragment.append(match[1]);
    else if (match[2]) span('tok-com', match[2]);
    else if (match[3]) {span(match[4] ? 'tok-key' : 'tok-str', match[3]); if (match[4]) fragment.append(match[4]);}
    else if (match[5]) span('tok-num', match[5]);
    else span('tok-kw', match[6]);
  }
  fragment.append(code.slice(last));
  return fragment;
}
function refreshCode() {
  $('code-view').replaceChildren(highlightCode(codeExamples[codeFormat]()));
  $('code-note').textContent = codeNote();
}
function selectCodeFormat(format) {
  codeFormat = format; storage.set('clef.codeFormat', format);
  document.querySelectorAll('.code-tabs [role=tab]').forEach(tab => {
    const selected = tab.dataset.format === format;
    tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1;
    if (selected) $('code-view').setAttribute('aria-labelledby', tab.id);
  });
  refreshCode();
}
async function copyCode() {
  const text = codeExamples[codeFormat](), button = $('copy-code');
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {
      // Plain-HTTP LAN addresses have no async clipboard API.
      const area = node('textarea'); area.value = text; area.setAttribute('readonly', ''); area.style.position = 'fixed'; area.style.opacity = '0';
      document.body.append(area); area.select(); const ok = document.execCommand('copy'); area.remove();
      if (!ok) throw new Error('copy failed');
    }
    button.textContent = 'Copied'; button.classList.add('done');
  } catch {button.textContent = 'Select and copy';}
  setTimeout(() => {button.textContent = 'Copy'; button.classList.remove('done');}, 1600);
}

/* Requests */
function apiError(status, payload) {
  if (status === 429) return payload?.detail || 'The request queue is full. Try again shortly.';
  if (status === 504) return payload?.detail || 'The request exceeded its queue wait deadline. Try again shortly.';
  if (status === 503) return payload?.detail || 'The GPU ran out of memory. Choose a smaller image scaling option, use fewer images, or shorten the context.';
  if (status === 413) return `${payload?.detail || 'This input exceeds the server limits.'}${Number.isFinite(payload?.input_tokens) ? ` Input: ${payload.input_tokens.toLocaleString()} tokens; accepted maximum: ${payload.max_input_tokens.toLocaleString()}.` : ''} Shorten the context, choose a smaller image scaling option, or use fewer images.`;
  if (Array.isArray(payload?.detail)) return payload.detail.map(error => error.msg).join(' ');
  return typeof payload?.detail === 'string' ? payload.detail : `Server returned HTTP ${status}. Try again or check the server connection.`;
}
async function submit(event) {
  event.preventDefault(); if (running || imageLoading) return;
  clearTimeout(autoRunTimer); autoRunPending = false;
  if (!apiModel) {await checkConnection(); if (!apiModel) {showError('Connect to the server before running a request.'); return;}}
  clearError();
  // Keep an answer that was typed but not yet added with Enter.
  for (const row of questionRows) if (row.type !== 'noul' && row.chipInput.value.trim() && addOptions(row, [row.chipInput.value], true)) row.chipInput.value = '';
  const questions = collectQuestions(true); if (!questions) return;
  const state = requestState();
  const context = $('shared-context').value.trim() || undefined;
  const imageScaling = $('image-scaling').value;
  const imagePooling = $('image-pooling').checked;
  running = true; $('editor-fields').disabled = true; $('example').disabled = true; $('run-button').disabled = true;
  $('run-button-text').textContent = images.length ? 'Preparing images…' : 'Answering…'; $('results').setAttribute('aria-busy', 'true');
  $('run-state').textContent = 'Answering'; $('run-state').className = 'run-state running';
  $('server-time').textContent = '—'; $('input-tokens').textContent = '—'; $('gpu-memory').textContent = '—';
  $('run-label').textContent = 'Total time includes image preparation, upload and response.'; $('timer-description').textContent = '';
  document.querySelectorAll('.answer-card').forEach(card => card.classList.add('stale'));
  if (!document.querySelector('.answer-card')) $('results').replaceChildren(node('div', 'empty-state', 'Clef is reading your input and answering your questions…'));
  $('result-announcement').textContent = images.length ? 'Preparing images for upload.' : 'Clef is answering.'; renderHistory();
  const run = {number: ++runSequence, timestamp: new Date().toISOString(), context, state, questions, imageNames: images.map(item => item.name), imageScaling, imagePooling, feedback: {}};
  const started = performance.now();
  function tick() {$('elapsed').textContent = ((performance.now() - started) / 1000).toFixed(3); elapsedFrame = requestAnimationFrame(tick);}
  tick();
  try {
    const prepared = [];
    for (const item of images) prepared.push(await ClefImages.prepare(item, imageScalings[imageScaling]));
    run.clientImageMs = performance.now() - started;
    const request = {model: apiModel, context, state, images: prepared.map(item => item.data), image_pooling: imagePooling, questions};
    if (prepared.length) request.media_kwargs = {images_kwargs: {do_resize: true, min_pixels: 1024, max_pixels: 20000000}};
    const body = JSON.stringify(request);
    run.uploadBytes = new TextEncoder().encode(body).byteLength;
    run.processedImages = prepared.map(({width, height, bytes, format}) => ({width, height, bytes, format}));
    $('run-button-text').textContent = 'Waiting for response…';
    $('result-announcement').textContent = 'Request sent. Clef will answer when its turn arrives.';
    const response = await fetch('/v1/systemone', {method: 'POST', headers: {'Content-Type': 'application/json'}, body, signal: AbortSignal.timeout(600000)});
    const responseText = await response.text();
    let payload;
    try {payload = JSON.parse(responseText);} catch {
      throw new Error(response.ok ? 'The server returned an unreadable response.' : apiError(response.status));
    }
    if (!response.ok) throw new Error(apiError(response.status, payload));
    if (!payload.answers || !payload.usage) throw new Error('The server returned an incomplete response.');
    run.response = payload;
  } catch (error) {
    run.error = error.name === 'TimeoutError' ? 'The request timed out after ten minutes. Check the server before retrying.' : error instanceof TypeError ? 'Could not reach the server. Check the connection, then try again.' : error.message;
  } finally {
    run.totalMs = performance.now() - started; cancelAnimationFrame(elapsedFrame);
    running = false; $('editor-fields').disabled = false; $('example').disabled = false; $('run-button').disabled = false;
    $('run-button-text').textContent = 'Ask Clef'; $('results').setAttribute('aria-busy', 'false');
    history.unshift(run); history = history.slice(0, 12); renderRun(run);
    void checkConnection();
    $('result-announcement').textContent = run.error ? `Request failed. ${run.error}` : `Answers received in ${duration(run.totalMs)}. Server processing took ${duration(run.response.usage.latency_ms)}.`;
    if (autoRunPending) {autoRunPending = false; scheduleAutoRun();}
  }
}

/* Wiring */
$('decision-form').addEventListener('submit', submit);
$('add-question').addEventListener('click', () => {addQuestion({}, true); markEdited(questionRows.at(-1).id);});
$('image-input').addEventListener('change', event => acceptImages(event.target.files));
document.addEventListener('paste', event => {
  const clipboard = event.clipboardData;
  if (!clipboard) return;
  let files = [...clipboard.items]
    .filter(item => item.kind === 'file' && item.type.startsWith('image/'))
    .map(item => item.getAsFile()).filter(Boolean);
  if (!files.length) files = [...clipboard.files].filter(file => file.type.startsWith('image/'));
  if (!files.length) return;
  event.preventDefault();
  if (running || imageLoading) {
    showError('Wait for the current request or image loading to finish, then paste again.');
    return;
  }
  const extensions = {'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp'};
  acceptImages(files.map(file => new File([file],
    `pasted-image-${++pastedImageSequence}.${extensions[file.type] || 'image'}`,
    {type: file.type})));
});
$('image-scaling').addEventListener('change', () => {updateScalingNote(); markEdited();});
$('image-pooling').addEventListener('change', () => markEdited());
$('drop-zone').addEventListener('dragover', event => {event.preventDefault(); if (!running) $('drop-zone').classList.add('dragging');});
$('drop-zone').addEventListener('dragleave', () => $('drop-zone').classList.remove('dragging'));
$('drop-zone').addEventListener('drop', event => {event.preventDefault(); $('drop-zone').classList.remove('dragging'); acceptImages(event.dataTransfer.files);});
$('connection').addEventListener('click', checkConnection);
for (const id of ['context', 'shared-context']) $(id).addEventListener('input', event => {fit(event.target); markEdited();});
$('shared-context').addEventListener('input', () => {
  const value = $('shared-context').value.trim();
  $('shared-peek').textContent = value || 'Optional instructions and definitions you reuse across requests';
});
$('example').addEventListener('click', () => {
  clearError(); questionRows = []; $('questions').replaceChildren(); images = []; renderImages(); $('image-input').value = '';
  $('context').value = 'Our checkout started returning errors and orders are blocked.'; fit($('context'));
  addQuestion({type: 'noul', instructions: 'Is a service down?'});
  addQuestion({type: 'choice', instructions: 'Which team should handle this message?', options: ['Billing', 'Technical support']});
  addQuestion({type: 'score', instructions: 'How urgent is this?', options: ['Can wait', 'This week', 'Today']});
  markEdited(); questionRows[0].text.focus();
});
$('clear-history').addEventListener('click', () => {if (!running) {history = []; renderHistory(); $('result-announcement').textContent = 'Run history cleared.';}});
$('auto-run').checked = storage.get('clef.autoRun') === '1';
$('auto-run').addEventListener('change', () => {storage.set('clef.autoRun', $('auto-run').checked ? '1' : '0'); if ($('auto-run').checked) scheduleAutoRun();});
$('show-code').addEventListener('click', () => {$('code-panel').scrollIntoView({block: 'start', behavior: 'smooth'}); $('code-view').focus({preventScroll: true});});
document.querySelectorAll('.code-tabs [role=tab]').forEach(tab => tab.addEventListener('click', () => selectCodeFormat(tab.dataset.format)));
$('code-panel').querySelector('.code-tabs').addEventListener('keydown', event => {
  if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
  const tabs = [...document.querySelectorAll('.code-tabs [role=tab]')];
  const next = tabs[(tabs.findIndex(tab => tab.dataset.format === codeFormat) + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
  selectCodeFormat(next.dataset.format); next.focus();
});
$('copy-code').addEventListener('click', copyCode);
$('questions').addEventListener('focusin', event => {const card = event.target.closest('.question-card'); if (card) highlight(card.dataset.link, true);});
document.querySelector('.bench').addEventListener('mouseover', event => {
  const item = event.target.closest('[data-link]'); if (item && !item.classList.contains('linked')) highlight(item.dataset.link, false);
});
document.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {event.preventDefault(); if (!running && !imageLoading) $('decision-form').requestSubmit();}
});
if (!isMac) $('run-shortcut').textContent = 'Ctrl+↵';
$('run-button').title = `Ask Clef (${isMac ? '⌘' : 'Ctrl'}+Enter)`;
addQuestion(); updateScalingNote();
selectCodeFormat(['curl', 'python', 'raw'].includes(storage.get('clef.codeFormat')) ? storage.get('clef.codeFormat') : 'curl');
checkConnection();
