/* ============================================================
   TicTacToe AI — frontend application
   ============================================================ */

'use strict';

// ── Global state ──────────────────────────────────────────────────────────
const G = {
  gameId:          null,
  mode:            'hva',   // 'hvh' | 'hva' | 'ava'
  playerSide:      'X',     // which side the human plays (hva only)
  agentOpponent:   'mcts_vanilla',
  agentX:          'mcts_vanilla',
  agentO:          'random',
  avaDelay:        500,
  sessionWins:     { X: 0, O: 0, draws: 0 },
  consecutiveWins: 0,
  lastWinner:      null,     // 'X' | 'O' | 'draw' | null
  isAiRunning:     false,
  avaLoopActive:   false,
  settings: {
    n: 3, k: 0,
    matchMode: 'time', timeLimitMs: 1000, nodeBudget: 100000, fixedDepth: 4,
    mctsC: 1.414, mctsRollout: 200,
  },
};

// ── API helpers ────────────────────────────────────────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}
const GET  = (path)       => api('GET',    path);
const POST = (path, body) => api('POST',   path, body);
const DEL  = (path)       => api('DELETE', path);

// ── Initialisation ─────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  await populateAgentSelects();
  await loadConfig();
  bindEvents();
  refreshHistory();
  refreshScores();
});

async function populateAgentSelects() {
  let agents = [];
  try { agents = await GET('/api/agents'); } catch (_) { return; }

  const byTier = {};
  for (const a of agents) {
    (byTier[a.tier] = byTier[a.tier] || []).push(a);
  }
  const tierLabel = { 0: 'Tier 0 — Baseline', 1: 'Tier 1 — Exact', 2: 'Tier 2 — Heuristic', 3: 'Tier 3 — MCTS' };

  function buildOptions(select, defaultId) {
    select.innerHTML = '';
    for (const tier of Object.keys(byTier).sort()) {
      const grp = document.createElement('optgroup');
      grp.label = tierLabel[tier] || `Tier ${tier}`;
      for (const a of byTier[tier]) {
        const opt = document.createElement('option');
        opt.value = a.id;
        opt.textContent = a.name;
        opt.title = a.description;
        if (a.id === defaultId) opt.selected = true;
        grp.appendChild(opt);
      }
      select.appendChild(grp);
    }
  }

  buildOptions($('sel-agent-opponent'), G.agentOpponent);
  buildOptions($('sel-agent-x'), G.agentX);
  buildOptions($('sel-agent-o'), G.agentO);
}

async function loadConfig() {
  try {
    const cfg = await GET('/api/config');
    if (cfg.budget) {
      G.settings.timeLimitMs  = cfg.budget.time_limit_ms ?? 1000;
      G.settings.nodeBudget   = cfg.budget.node_budget   ?? 100000;
      G.settings.fixedDepth   = cfg.budget.fixed_depth   ?? 4;
    }
    if (cfg.mcts) {
      G.settings.mctsC        = cfg.mcts.exploration_constant ?? 1.414;
      G.settings.mctsRollout  = cfg.mcts.rollout_depth_limit  ?? 200;
    }
    if (cfg.game) {
      G.settings.n = cfg.game.n ?? 3;
      G.settings.k = cfg.game.k ?? 0;
    }
    syncSettingsInputs();
  } catch (_) {}
}

// ── Event binding ──────────────────────────────────────────────────────────
function bindEvents() {
  // Mode tabs
  qAll('.mode-tab').forEach(btn =>
    btn.addEventListener('click', () => {
      qAll('.mode-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      G.mode = btn.dataset.mode;
      showModeCfg(G.mode);
    })
  );

  // Player-side toggle (hva)
  qAll('.toggle-btn').forEach(btn =>
    btn.addEventListener('click', () => {
      qAll('.toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      G.playerSide = btn.dataset.side;
    })
  );

  // Agent selects
  $('sel-agent-opponent').addEventListener('change', e => G.agentOpponent = e.target.value);
  $('sel-agent-x').addEventListener('change', e => G.agentX = e.target.value);
  $('sel-agent-o').addEventListener('change', e => G.agentO = e.target.value);

  // AvA delay slider
  const delaySlider = $('ava-delay');
  delaySlider.addEventListener('input', () => {
    G.avaDelay = parseInt(delaySlider.value, 10);
    $('ava-delay-label').textContent = `${G.avaDelay} ms`;
  });

  // New game / resign
  $('btn-new-game').addEventListener('click', startNewGame);
  $('btn-resign').addEventListener('click', resignGame);

  // Win/draw overlay play-again buttons
  $('btn-win-new').addEventListener('click', () => { hideOverlay('win-overlay'); startNewGame(); });
  $('btn-draw-new').addEventListener('click', () => { hideOverlay('draw-overlay'); startNewGame(); });

  // Settings
  $('btn-settings').addEventListener('click', openSettings);
  $('btn-settings-close').addEventListener('click', closeSettings);
  $('btn-settings-save').addEventListener('click', saveSettings);
  $('btn-settings-reset').addEventListener('click', resetSettings);
  $('settings-modal').addEventListener('click', e => { if (e.target === $('settings-modal')) closeSettings(); });

  $('set-match-mode').addEventListener('change', () => updateBudgetRows());

  // History refresh
  $('btn-refresh-history').addEventListener('click', refreshHistory);
}

// ── Mode config panel ──────────────────────────────────────────────────────
function showModeCfg(mode) {
  $('cfg-hva').classList.toggle('hidden', mode !== 'hva');
  $('cfg-hvh').classList.toggle('hidden', mode !== 'hvh');
  $('cfg-ava').classList.toggle('hidden', mode !== 'ava');
}

// ── New game ───────────────────────────────────────────────────────────────
async function startNewGame() {
  stopAvaLoop();
  if (G.gameId) {
    try { await DEL(`/api/game/${G.gameId}`); } catch (_) {}
    G.gameId = null;
  }

  const { settings: s } = G;

  // Determine which agent IDs to send
  let agentX, agentO;
  if (G.mode === 'hva') {
    agentX = G.playerSide === 'X' ? 'random' : G.agentOpponent; // 'random' is placeholder for human
    agentO = G.playerSide === 'O' ? 'random' : G.agentOpponent;
  } else {
    agentX = G.agentX;
    agentO = G.agentO;
  }

  const body = {
    mode:          G.mode,
    player_side:   G.playerSide,
    agent_x:       agentX,
    agent_o:       agentO,
    n:             s.n,
    k:             s.k,
    match_mode:    s.matchMode,
    time_limit_ms: s.timeLimitMs,
    node_budget:   s.nodeBudget,
    fixed_depth:   s.fixedDepth,
  };

  let resp;
  try {
    resp = await POST('/api/game/new', body);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
    return;
  }

  G.gameId = resp.game_id;
  $('btn-resign').disabled = false;
  renderAll(resp);

  // If it's AvA or it's HvA and the human plays O (AI goes first as X)
  if (G.mode === 'ava') {
    startAvaLoop();
  } else if (G.mode === 'hva' && G.playerSide === 'O') {
    await doAiStep(); // AI (X) goes first
  }
}

// ── Human move ─────────────────────────────────────────────────────────────
async function onCellClick(row, col) {
  if (!G.gameId || G.isAiRunning) return;

  let resp;
  try {
    resp = await POST(`/api/game/${G.gameId}/move`, { row, col });
  } catch (err) {
    setStatus(`${err.message}`);
    return;
  }

  renderAll(resp);
  if (resp.game_over) return;

  // In HvA mode, trigger AI reply
  if (G.mode === 'hva') {
    await doAiStep();
  }
}

// ── AI step ────────────────────────────────────────────────────────────────
async function doAiStep() {
  if (!G.gameId) return;
  G.isAiRunning = true;
  showThinking(true);

  let resp;
  try {
    resp = await POST(`/api/game/${G.gameId}/ai-step`);
  } catch (err) {
    setStatus(`AI error: ${err.message}`);
    G.isAiRunning = false;
    showThinking(false);
    return;
  }

  G.isAiRunning = false;
  showThinking(false);
  renderAll(resp);
}

// ── AvA loop ───────────────────────────────────────────────────────────────
function startAvaLoop() {
  G.avaLoopActive = true;
  scheduleAvaStep();
}

function stopAvaLoop() {
  G.avaLoopActive = false;
}

function scheduleAvaStep() {
  if (!G.avaLoopActive || !G.gameId) return;
  setTimeout(async () => {
    if (!G.avaLoopActive || !G.gameId) return;
    await doAiStep();
    if (G.avaLoopActive && G.gameId) scheduleAvaStep();
  }, G.avaDelay);
}

// ── Resign ─────────────────────────────────────────────────────────────────
async function resignGame() {
  stopAvaLoop();
  if (!G.gameId) return;
  try { await DEL(`/api/game/${G.gameId}`); } catch (_) {}
  G.gameId = null;
  $('btn-resign').disabled = true;
  setStatus('Game abandoned.');
  buildBoard([], 0, null);
  updateProbabilityBar(0.5);
}

// ── Render helpers ─────────────────────────────────────────────────────────
function renderAll(resp) {
  const { state, probability_x, game_over, result,
          player_x_name, player_o_name } = resp;

  buildBoard(state.board, state.n, state.last_move, state.result);
  updateProbabilityBar(probability_x, player_x_name, player_o_name);

  if (game_over) {
    G.gameId = null;
    $('btn-resign').disabled = true;
    handleGameOver(result, player_x_name, player_o_name);
    refreshHistory();
    refreshScores();
  } else {
    const whose = state.current_player === 'X'
      ? `<span class="turn-x">${player_x_name || 'X'}</span>`
      : `<span class="turn-o">${player_o_name || 'O'}</span>`;
    setStatus(`${whose}'s turn`);
  }
}

function buildBoard(board, n, lastMove, result) {
  const boardEl = $('board');
  boardEl.style.gridTemplateColumns = `repeat(${n || 3}, 1fr)`;

  const prev = boardEl.querySelectorAll('.cell');
  const oldPieces = {};
  prev.forEach(c => {
    const key = `${c.dataset.row},${c.dataset.col}`;
    oldPieces[key] = c.dataset.piece;
  });

  boardEl.innerHTML = '';
  if (!board || board.length === 0) return;

  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.row = r;
      cell.dataset.col = c;

      const piece = board[r][c]; // 'X' | 'O' | 'EMPTY'
      cell.dataset.piece = piece;

      if (piece === 'X') {
        cell.textContent = 'X';
        cell.classList.add('cell-x', 'cell-taken');
        if (oldPieces[`${r},${c}`] !== 'X') cell.classList.add('cell-new');
      } else if (piece === 'O') {
        cell.textContent = 'O';
        cell.classList.add('cell-o', 'cell-taken');
        if (oldPieces[`${r},${c}`] !== 'O') cell.classList.add('cell-new');
      } else {
        cell.addEventListener('click', () => onCellClick(r, c));
      }

      if (lastMove && lastMove[0] === r && lastMove[1] === c) {
        cell.classList.add('cell-last');
      }

      boardEl.appendChild(cell);
    }
  }
}

function highlightWinningCells(board, n, result) {
  if (!result || result === 'IN_PROGRESS' || result === 'DRAW') return;
  const winPiece = result === 'X_WINS' ? 'X' : 'O';
  document.querySelectorAll('.cell').forEach(cell => {
    if (cell.dataset.piece === winPiece) {
      cell.classList.add('cell-win');
    }
  });
}

function updateProbabilityBar(probX, playerX, playerO) {
  const pct = Math.round(probX * 100);
  $('prob-fill').style.width = `${pct}%`;
  $('prob-x-label').textContent = `${playerX || 'X'} ${pct}%`;
  $('prob-o-label').textContent = `${playerO || 'O'} ${100 - pct}%`;
}

function setStatus(html) {
  $('game-status').innerHTML = html;
}

function showThinking(show) {
  $('thinking-bar').classList.toggle('hidden', !show);
  if (show) {
    const curr = document.querySelector('.cell-last');
    const player = document.querySelector('.turn-x') ? 'O' : 'X';
    $('thinking-label').textContent = `${player} is thinking…`;
  }
}

// ── Game-over handling ─────────────────────────────────────────────────────
function handleGameOver(result, playerX, playerO) {
  stopAvaLoop();

  // Update session scores
  const prevWinner = G.lastWinner;
  if (result === 'X_WINS') {
    G.sessionWins.X++;
    G.lastWinner = 'X';
  } else if (result === 'O_WINS') {
    G.sessionWins.O++;
    G.lastWinner = 'O';
  } else {
    G.sessionWins.draws++;
    G.lastWinner = 'draw';
  }

  // Consecutive-win tracking (only for non-draws)
  if (result !== 'DRAW') {
    if (G.lastWinner === prevWinner) {
      G.consecutiveWins++;
    } else {
      G.consecutiveWins = 1;
    }
  } else {
    G.consecutiveWins = 0;
  }

  animateScoreRoll();
  updateStreakBadge();

  // Highlight winning cells
  const board = [];
  document.querySelectorAll('.cell').forEach(c => {
    const r = parseInt(c.dataset.row, 10);
    const co = parseInt(c.dataset.col, 10);
    if (!board[r]) board[r] = [];
    board[r][co] = c.dataset.piece;
  });
  highlightWinningCells(board, Math.sqrt(document.querySelectorAll('.cell').length), result);

  // Show overlay
  if (result === 'DRAW') {
    showOverlay('draw-overlay');
    setStatus('Draw!');
  } else {
    const winner = result === 'X_WINS' ? (playerX || 'X') : (playerO || 'O');
    $('win-title').textContent = `${winner} Wins!`;
    $('win-icon').textContent = G.consecutiveWins >= 5 ? '🔥' : G.consecutiveWins >= 3 ? '⚡' : '🏆';
    $('win-streak').textContent = G.consecutiveWins >= 2 ? `${G.consecutiveWins}× WIN STREAK!` : '';
    showOverlay('win-overlay');
    setStatus(`<strong>${winner}</strong> wins!`);
    launchConfetti(G.consecutiveWins);

    if (G.consecutiveWins >= 3) {
      document.body.classList.add('shake');
      setTimeout(() => document.body.classList.remove('shake'), 500);
    }
  }
}

// ── Score counter slot-machine animation ──────────────────────────────────
function animateScoreRoll() {
  const xEl = $('score-x');
  const oEl = $('score-o');
  const dEl = $('score-draws');

  xEl.textContent = G.sessionWins.X;
  oEl.textContent = G.sessionWins.O;
  dEl.textContent = G.sessionWins.draws;

  [xEl, oEl, dEl].forEach(el => {
    el.classList.remove('score-roll');
    void el.offsetWidth; // reflow to restart animation
    el.classList.add('score-roll');
    el.addEventListener('animationend', () => el.classList.remove('score-roll'), { once: true });
  });
}

function updateStreakBadge() {
  const badge = $('streak-badge');
  if (G.consecutiveWins >= 2) {
    badge.classList.remove('hidden');
    const icons = ['', '', '🔥', '⚡', '💥', '🌊', '🚀'];
    const icon = icons[Math.min(G.consecutiveWins, icons.length - 1)] || '🔥';
    badge.textContent = `${icon} ${G.consecutiveWins}× STREAK`;
  } else {
    badge.classList.add('hidden');
  }
}

// ── Confetti ───────────────────────────────────────────────────────────────
function launchConfetti(streak) {
  const canvas = $('confetti-canvas');
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  const ctx = canvas.getContext('2d');

  const count = Math.min(60 + streak * 30, 200);
  const particles = Array.from({ length: count }, () => ({
    x: Math.random() * canvas.width,
    y: -20,
    vx: (Math.random() - 0.5) * 4,
    vy: Math.random() * 4 + 2,
    rot: Math.random() * 360,
    rotV: (Math.random() - 0.5) * 8,
    size: Math.random() * 8 + 4,
    color: ['#00c74d', '#e63946', '#ffd700', '#4fc3f7', '#ff8a65'][Math.floor(Math.random() * 5)],
    life: 1.0,
  }));

  let frame;
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let alive = false;
    for (const p of particles) {
      p.x   += p.vx;
      p.y   += p.vy;
      p.rot += p.rotV;
      p.vy  += 0.12;
      p.life -= 0.008;
      if (p.life <= 0) continue;
      alive = true;
      ctx.save();
      ctx.globalAlpha = Math.max(0, p.life);
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot * Math.PI / 180);
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
      ctx.restore();
    }
    if (alive) frame = requestAnimationFrame(draw);
    else ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  if (frame) cancelAnimationFrame(frame);
  draw();
}

// ── Overlay helpers ────────────────────────────────────────────────────────
function showOverlay(id) {
  $(id).classList.remove('hidden');
  if (id === 'draw-overlay') $('confetti-canvas').getContext('2d').clearRect(0, 0, 9999, 9999);
}
function hideOverlay(id) { $(id).classList.add('hidden'); }

// ── Settings modal ─────────────────────────────────────────────────────────
function openSettings() {
  syncSettingsInputs();
  $('settings-modal').classList.remove('hidden');
}
function closeSettings() { $('settings-modal').classList.add('hidden'); }

function syncSettingsInputs() {
  const s = G.settings;
  $('set-n').value         = s.n;
  $('set-k').value         = s.k;
  $('set-match-mode').value = s.matchMode;
  $('set-time').value       = s.timeLimitMs;
  $('set-nodes').value      = s.nodeBudget;
  $('set-depth').value      = s.fixedDepth;
  $('set-mcts-c').value     = s.mctsC;
  $('set-mcts-rollout').value = s.mctsRollout;
  updateBudgetRows();
}

function updateBudgetRows() {
  const mode = $('set-match-mode').value;
  $('set-row-time').classList.toggle('hidden',  mode !== 'time');
  $('set-row-node').classList.toggle('hidden',  mode !== 'node');
  $('set-row-depth').classList.toggle('hidden', mode !== 'depth');
}

function saveSettings() {
  const s = G.settings;
  s.n           = Math.max(3, Math.min(15, parseInt($('set-n').value, 10) || 3));
  s.k           = Math.max(0, Math.min(15, parseInt($('set-k').value, 10) || 0));
  s.matchMode   = $('set-match-mode').value;
  s.timeLimitMs = Math.max(50, parseFloat($('set-time').value) || 1000);
  s.nodeBudget  = Math.max(1000, parseInt($('set-nodes').value, 10) || 100000);
  s.fixedDepth  = Math.max(1, Math.min(20, parseInt($('set-depth').value, 10) || 4));
  s.mctsC       = Math.max(0, parseFloat($('set-mcts-c').value) || 1.414);
  s.mctsRollout = Math.max(1, parseInt($('set-mcts-rollout').value, 10) || 200);
  closeSettings();
}

function resetSettings() {
  G.settings = { n: 3, k: 0, matchMode: 'time', timeLimitMs: 1000, nodeBudget: 100000, fixedDepth: 4, mctsC: 1.414, mctsRollout: 200 };
  syncSettingsInputs();
}

// ── History & leaderboard ─────────────────────────────────────────────────
async function refreshHistory() {
  try {
    const rows = await GET('/api/history?limit=30');
    renderHistory(rows);
  } catch (_) {}
}

async function refreshScores() {
  try {
    const rows = await GET('/api/scores');
    renderScores(rows);
  } catch (_) {}
}

function renderHistory(rows) {
  const el = $('history-list');
  if (!rows.length) {
    el.innerHTML = '<p class="empty-msg">No games recorded yet.</p>';
    return;
  }
  el.innerHTML = rows.map(r => {
    const resultClass = r.result === 'X_WINS' ? 'result-x' : r.result === 'O_WINS' ? 'result-o' : 'result-draw';
    const resultText  = r.result === 'X_WINS' ? `${r.player_x} won` : r.result === 'O_WINS' ? `${r.player_o} won` : 'Draw';
    const board = `${r.n}×${r.n}${r.k !== r.n ? ` k=${r.k}` : ''}`;
    return `<div class="history-item">
      <span class="history-players">${r.player_x} vs ${r.player_o} <span class="muted">(${board})</span></span>
      <span class="history-result ${resultClass}">${resultText}</span>
    </div>`;
  }).join('');
}

function renderScores(rows) {
  const el = $('scores-table');
  if (!rows.length) {
    el.innerHTML = '<p class="empty-msg">No data yet.</p>';
    return;
  }
  el.innerHTML = `
    <div class="scores-row scores-header">
      <span class="scores-name">Player</span>
      <span class="scores-w">W</span>
      <span class="scores-l">L</span>
      <span class="scores-d">D</span>
    </div>
  ` + rows.map(r =>
    `<div class="scores-row">
      <span class="scores-name" title="${r.player_name}">${r.player_name}</span>
      <span class="scores-w">${r.wins}</span>
      <span class="scores-l">${r.losses}</span>
      <span class="scores-d">${r.draws}</span>
    </div>`
  ).join('');
}

// ── DOM helpers ────────────────────────────────────────────────────────────
function  $(id) { return document.getElementById(id); }
function qAll(sel) { return document.querySelectorAll(sel); }
