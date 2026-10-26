/**
 * Hidani AI Job Assistant - Pure Vanilla JS Conversational Engine
 * Handles chat state, resume document upload, API communication,
 * and dynamic rendering of interactive job cards and sub-agent widgets.
 */

// Configure API Base URL: auto-adapts to local dev or production proxy
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000/api'
  : '/api';

// Application State
const state = {
  sessionId: localStorage.getItem('hidani_session_id') || `sess_${Math.random().toString(36).substring(2, 10)}`,
  profile: JSON.parse(localStorage.getItem('hidani_candidate_profile') || 'null'),
  isSending: false,
  appliedJobIds: new Set(JSON.parse(localStorage.getItem('hidani_applied_jobs') || '[]'))
};

// Persist session ID
localStorage.setItem('hidani_session_id', state.sessionId);

// DOM Elements
const elements = {
  chatScrollArea: document.getElementById('chatScrollArea'),
  messagesContainer: document.getElementById('messagesContainer'),
  chatForm: document.getElementById('chatForm'),
  messageInput: document.getElementById('messageInput'),
  sendBtn: document.getElementById('sendBtn'),
  fileInput: document.getElementById('fileInput'),
  attachBtn: document.getElementById('attachBtn'),
  resetChatBtn: document.getElementById('resetChatBtn'),
  dropOverlay: document.getElementById('dropOverlay'),
  candidatePillBtn: document.getElementById('candidatePillBtn'),
  candidateInitial: document.getElementById('candidateInitial'),
  candidateNameBadge: document.getElementById('candidateNameBadge'),
  candidateSkillsCount: document.getElementById('candidateSkillsCount'),
  jobsCountLabel: document.getElementById('jobsCountLabel'),
  profileDrawer: document.getElementById('profileDrawer'),
  closeDrawerBtn: document.getElementById('closeDrawerBtn'),
  drawerAvatar: document.getElementById('drawerAvatar'),
  drawerName: document.getElementById('drawerName'),
  drawerEmail: document.getElementById('drawerEmail'),
  drawerExpLevel: document.getElementById('drawerExpLevel'),
  drawerExpYears: document.getElementById('drawerExpYears'),
  drawerSkillsTags: document.getElementById('drawerSkillsTags'),
  drawerSummary: document.getElementById('drawerSummary'),
  drawerRoles: document.getElementById('drawerRoles'),
  toastContainer: document.getElementById('toastContainer')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  updateProfileUI();
  fetchSystemHealth();
  elements.messageInput.focus();
});

/**
 * Event Listeners Registration
 */
function setupEventListeners() {
  // Chat form submit
  elements.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    handleSendMessage();
  });

  // Enter to send, Shift+Enter for new line
  elements.messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });

  // Auto-resize input
  elements.messageInput.addEventListener('input', () => {
    elements.messageInput.style.height = 'auto';
    elements.messageInput.style.height = Math.min(elements.messageInput.scrollHeight, 120) + 'px';
  });

  // File upload triggers
  elements.attachBtn.addEventListener('click', () => elements.fileInput.click());
  elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Drag and Drop
  window.addEventListener('dragenter', (e) => {
    e.preventDefault();
    elements.dropOverlay.classList.remove('hidden');
    elements.dropOverlay.classList.add('flex');
  });

  elements.dropOverlay.addEventListener('dragover', (e) => e.preventDefault());
  elements.dropOverlay.addEventListener('dragleave', (e) => {
    if (e.relatedTarget === null) {
      elements.dropOverlay.classList.add('hidden');
      elements.dropOverlay.classList.remove('flex');
    }
  });

  window.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.dropOverlay.classList.add('hidden');
    elements.dropOverlay.classList.remove('flex');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  // Profile Drawer toggle
  elements.candidatePillBtn.addEventListener('click', () => openProfileDrawer());
  elements.closeDrawerBtn.addEventListener('click', () => closeProfileDrawer());
  elements.profileDrawer.addEventListener('click', (e) => {
    if (e.target === elements.profileDrawer) closeProfileDrawer();
  });

  // Reset Session
  elements.resetChatBtn.addEventListener('click', () => resetSession());

  // Global helper for quick prompts
  window.sendPrompt = (promptText) => {
    elements.messageInput.value = promptText;
    handleSendMessage();
  };

  // Global helper for one-click job applying
  window.applyToJob = (jobId) => {
    elements.messageInput.value = `apply to ${jobId}`;
    handleSendMessage();
  };
}

/**
 * Send user message to Orchestrator API
 */
async function handleSendMessage() {
  const text = elements.messageInput.value.trim();
  if (!text || state.isSending) return;

  // Clear input
  elements.messageInput.value = '';
  elements.messageInput.style.height = 'auto';

  // Render User Message
  appendUserMessage(text);
  setSending(true);

  // Show typing indicator
  const typingIndicatorId = showTypingIndicator();
  scrollToBottom();

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        session_id: state.sessionId,
        profile: state.profile
      })
    });

    removeTypingIndicator(typingIndicatorId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server responded with status ${res.status}`);
    }

    const data = await res.json();
    appendAssistantMessage(data.reply, data.actions);

  } catch (error) {
    removeTypingIndicator(typingIndicatorId);
    appendSystemAlert(`Error contacting assistant: ${error.message}. Please verify the backend is running.`);
    showToast(`Error: ${error.message}`, 'error');
  } finally {
    setSending(false);
    scrollToBottom();
  }
}

/**
 * Handle Resume File Upload (PDF or DOCX)
 */
async function handleFileUpload(file) {
  const validExtensions = ['.pdf', '.docx'];
  const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();

  if (!validExtensions.includes(ext)) {
    showToast('Only .pdf and .docx documents are supported', 'error');
    return;
  }

  // Activity message
  const uploadActivityId = appendActivityMessage(`Analyzing **${file.name}** with Sub-Agent 1 (Parser Agent)...`);
  scrollToBottom();

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE}/upload-resume`, {
      method: 'POST',
      body: formData
    });

    const activityEl = document.getElementById(uploadActivityId);
    if (activityEl) activityEl.remove();

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Upload failed with status ${res.status}`);
    }

    const data = await res.json();
    const profile = data.profile;

    // Save profile into state
    state.profile = profile;
    localStorage.setItem('hidani_candidate_profile', JSON.stringify(profile));
    updateProfileUI();

    // Format all 7 required assessment fields
    const skillsFormatted = (profile.skills || []).join(', ') || 'Software Engineering, Python, REST APIs';
    const eduFormatted = (profile.education || []).join(' • ') || 'Bachelor\'s in Computer Science / Engineering';
    const projFormatted = (profile.projects || []).map(p => `  • ${p}`).join('\n') || '  • Software engineering and web development projects';
    const contactInfo = profile.contact || [profile.email, profile.phone].filter(Boolean).join(' | ') || 'Contact provided in resume';

    const parseSummaryMarkdown = (
      `### 📄 Resume Parsed Successfully!\n\n` +
      `Here are your extracted profile details:\n\n` +
      `| Field | Extracted Details |\n` +
      `| :--- | :--- |\n` +
      `| 👤 **Name** | **${profile.name || 'Candidate'}** |\n` +
      `| 📞 **Contact** | ${contactInfo} |\n` +
      `| 💼 **Designation** | **${profile.designation || 'Software Engineer'}** |\n` +
      `| 🛠️ **Skills** | ${skillsFormatted} |\n` +
      `| ⏱️ **Experience** | ${profile.experience || profile.experience_years || '1-3 years'} (${profile.experience_level || 'Mid-Level'}) |\n` +
      `| 🎓 **Education** | ${eduFormatted} |\n\n` +
      `**🚀 Key Projects:**\n${projFormatted}\n\n` +
      `*Finding top matching positions for your designation, skills, and experience...*`
    );

    appendAssistantMessage(parseSummaryMarkdown);
    showToast(`Parsed resume for ${profile.name || 'Candidate'}!`, 'success');

    // Auto-trigger Step 3: Recommend jobs matching designation, skills, and experience
    const recActivityId = appendActivityMessage('Matching jobs against 1,600+ database listings with Sub-Agent 2 (Matcher)...');
    scrollToBottom();

    try {
      const recRes = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: `Recommend jobs matching my designation (${profile.designation}), skills (${(profile.skills || []).slice(0, 5).join(', ')}), and experience (${profile.experience_level})`,
          session_id: state.sessionId,
          profile: profile
        })
      });

      const recActivityEl = document.getElementById(recActivityId);
      if (recActivityEl) recActivityEl.remove();

      if (recRes.ok) {
        const recData = await recRes.json();
        appendAssistantMessage(recData.reply, recData.actions);
      }
    } catch (recErr) {
      const recActivityEl = document.getElementById(recActivityId);
      if (recActivityEl) recActivityEl.remove();
      console.warn('Auto-recommendation failed:', recErr);
    }

  } catch (error) {
    const activityEl = document.getElementById(uploadActivityId);
    if (activityEl) activityEl.remove();
    appendSystemAlert(`Document Parsing Failed: ${error.message}`);
    showToast(`Parsing failed: ${error.message}`, 'error');
  } finally {
    elements.fileInput.value = '';
    scrollToBottom();
  }
}

/**
 * Message Renderers
 */
function appendUserMessage(text) {
  const msgEl = document.createElement('div');
  msgEl.className = 'flex justify-end message-appear';
  msgEl.innerHTML = `
    <div class="max-w-[85%] sm:max-w-[75%] bg-brand-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 shadow-md text-sm leading-relaxed">
      <p class="whitespace-pre-wrap">${escapeHtml(text)}</p>
    </div>
  `;
  elements.messagesContainer.appendChild(msgEl);
}

function appendAssistantMessage(markdownContent, actions = []) {
  const msgEl = document.createElement('div');
  msgEl.className = 'flex items-start gap-3 message-appear';

  // Parse Markdown with marked.js
  const parsedHtml = marked.parse(markdownContent || '');

  msgEl.innerHTML = `
    <div class="w-8 h-8 rounded-xl bg-gradient-to-tr from-brand-600 via-indigo-600 to-cyan-500 text-white flex items-center justify-center flex-shrink-0 shadow-md shadow-brand-500/20 mt-1">
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    </div>
    <div class="flex-1 max-w-[90%] sm:max-w-[85%] bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-sm p-4 shadow-lg text-slate-200 text-sm">
      <div class="prose-chat">${parsedHtml}</div>
      <div class="actions-slot mt-3 space-y-3"></div>
    </div>
  `;

  elements.messagesContainer.appendChild(msgEl);

  // Render structured widgets if actions were returned
  const actionsSlot = msgEl.querySelector('.actions-slot');
  if (actions && actions.length > 0) {
    actions.forEach(action => {
      if (action.type === 'RECOMMEND_JOBS' && Array.isArray(action.data)) {
        renderInteractiveJobCards(actionsSlot, action.data);
      } else if (action.type === 'APPLICATION_SUBMITTED' && action.data) {
        renderApplicationBadge(actionsSlot, action.data);
      }
    });
  }
}

/**
 * Render Interactive Job Cards inside the Assistant's message
 */
function renderInteractiveJobCards(container, jobs) {
  if (!jobs || jobs.length === 0) return;

  const cardList = document.createElement('div');
  cardList.className = 'space-y-3 pt-2';

  jobs.forEach((job, idx) => {
    const isApplied = state.appliedJobIds.has(job.job_id);
    const score = job.match_score || 85;

    const card = document.createElement('div');
    card.className = 'p-4 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-slate-700 transition-all shadow-sm';
    
    // Skills tags pills
    const skillPills = (job.skills || []).slice(0, 4).map(s => 
      `<span class="px-2 py-0.5 rounded-md bg-slate-800 text-[10px] text-slate-300 font-mono">${escapeHtml(s)}</span>`
    ).join(' ');

    card.innerHTML = `
      <div class="flex items-start justify-between gap-3 mb-2">
        <div class="flex items-start gap-2.5">
          <span class="px-2 py-1 rounded-lg bg-brand-500/20 text-brand-300 font-mono font-bold text-xs border border-brand-500/30 flex-shrink-0">#${idx + 1}</span>
          <div>
            <h4 class="text-sm font-bold text-white">${escapeHtml(job.title)}</h4>
            <p class="text-xs text-slate-400 font-medium">${escapeHtml(job.company)} • <span class="text-slate-300">${escapeHtml(job.location)}</span></p>
          </div>
        </div>
        <div class="text-right flex-shrink-0">
          <div class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 text-xs font-mono font-bold">
            <span>${score}%</span>
            <span class="text-[9px] font-sans font-normal text-emerald-300">Match</span>
          </div>
        </div>
      </div>

      <!-- Match Reason Highlight -->
      <div class="text-xs text-slate-300 bg-slate-900/90 rounded-lg p-2.5 mb-3 border border-slate-800/80 leading-relaxed">
        <span class="text-brand-400 font-semibold text-[11px] block mb-0.5">💡 Why you fit:</span>
        ${escapeHtml(job.match_reason || 'Strong alignment with candidate technical skills.')}
      </div>

      <!-- Skills and Badges -->
      <div class="flex flex-wrap items-center gap-1.5 mb-3.5">
        <span class="px-2 py-0.5 rounded-md bg-indigo-500/15 text-indigo-300 border border-indigo-500/20 text-[10px] font-semibold">${escapeHtml(job.experience || 'Mid-Level')}</span>
        ${skillPills}
      </div>

      <!-- Action Buttons -->
      <div class="flex items-center justify-between pt-2 border-t border-slate-800/80">
        <span class="font-mono text-[10px] text-slate-500">ID: ${escapeHtml(job.job_id)}</span>
        <div class="flex items-center gap-2">
          ${job.source_link ? `
            <a href="${escapeHtml(job.source_link)}" target="_blank" rel="noopener noreferrer" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition-colors inline-flex items-center gap-1">
              <span>View</span>
              <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" /></svg>
            </a>
          ` : ''}
          <button
            onclick="window.applyToJob('${escapeHtml(job.job_id)}')"
            ${isApplied ? 'disabled' : ''}
            class="px-3.5 py-1.5 rounded-lg ${isApplied ? 'bg-slate-800 text-slate-500 cursor-not-allowed' : 'bg-brand-600 hover:bg-brand-500 text-white shadow shadow-brand-600/30'} text-xs font-semibold transition-all inline-flex items-center gap-1.5"
          >
            ${isApplied ? `
              <svg class="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>
              <span>Applied</span>
            ` : `
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" /></svg>
              <span>Quick Apply</span>
            `}
          </button>
        </div>
      </div>
    `;

    cardList.appendChild(card);
  });

  container.appendChild(cardList);
}

/**
 * Render application confirmation widget
 */
function renderApplicationBadge(container, appData) {
  if (!appData) return;
  const badge = document.createElement('div');
  const isSuccess = appData.success;
  
  if (isSuccess && appData.application) {
    state.appliedJobIds.add(appData.application.job_id);
    localStorage.setItem('hidani_applied_jobs', JSON.stringify(Array.from(state.appliedJobIds)));
  }

  badge.className = `p-3.5 rounded-xl border ${isSuccess ? 'bg-emerald-950/30 border-emerald-500/40 text-emerald-200' : 'bg-amber-950/30 border-amber-500/40 text-amber-200'} text-xs`;
  badge.innerHTML = `
    <div class="flex items-center gap-2">
      <span class="w-2 h-2 rounded-full ${isSuccess ? 'bg-emerald-400' : 'bg-amber-400'}"></span>
      <span class="font-bold">${isSuccess ? 'Application Saved to Database' : 'Application Notice'}</span>
    </div>
    <p class="mt-1 text-slate-300">${escapeHtml(appData.message)}</p>
  `;
  container.appendChild(badge);
}

function appendActionChip(label, onClick) {
  const chipContainer = document.createElement('div');
  chipContainer.className = 'mt-3';
  const chip = document.createElement('button');
  chip.className = 'px-3.5 py-1.5 rounded-xl bg-brand-600/20 hover:bg-brand-600/30 text-brand-300 border border-brand-500/40 text-xs font-medium transition-all';
  chip.innerText = label;
  chip.onclick = onClick;
  chipContainer.appendChild(chip);
  elements.messagesContainer.appendChild(chipContainer);
}

function appendActivityMessage(text) {
  const id = `act_${Date.now()}`;
  const el = document.createElement('div');
  el.id = id;
  el.className = 'flex items-center gap-2.5 px-4 py-2.5 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-300 message-appear w-fit';
  el.innerHTML = `
    <div class="w-4 h-4 border-2 border-brand-500 border-t-transparent rounded-full animate-spin"></div>
    <span>${marked.parseInline(text)}</span>
  `;
  elements.messagesContainer.appendChild(el);
  return id;
}

function appendSystemAlert(text) {
  const el = document.createElement('div');
  el.className = 'p-3 rounded-xl bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 message-appear';
  el.innerHTML = `<strong>Notice:</strong> ${escapeHtml(text)}`;
  elements.messagesContainer.appendChild(el);
}

/**
 * Typing Indicator with Bouncing Dots
 */
function showTypingIndicator() {
  const id = `typing_${Date.now()}`;
  const el = document.createElement('div');
  el.id = id;
  el.className = 'flex items-center gap-3 message-appear';
  el.innerHTML = `
    <div class="w-8 h-8 rounded-xl bg-slate-800 text-slate-400 flex items-center justify-center flex-shrink-0">
      <svg class="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
      </svg>
    </div>
    <div class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-sm px-4 py-3 shadow-md flex items-center gap-1.5">
      <span class="w-1.5 h-1.5 rounded-full bg-brand-400 animate-bounce" style="animation-delay: 0ms"></span>
      <span class="w-1.5 h-1.5 rounded-full bg-brand-400 animate-bounce" style="animation-delay: 150ms"></span>
      <span class="w-1.5 h-1.5 rounded-full bg-brand-400 animate-bounce" style="animation-delay: 300ms"></span>
    </div>
  `;
  elements.messagesContainer.appendChild(el);
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

/**
 * UI State Helpers
 */
function setSending(isSending) {
  state.isSending = isSending;
  elements.sendBtn.disabled = isSending;
}

function scrollToBottom() {
  elements.chatScrollArea.scrollTo({
    top: elements.chatScrollArea.scrollHeight,
    behavior: 'smooth'
  });
}

function updateProfileUI() {
  if (state.profile) {
    elements.candidatePillBtn.classList.remove('hidden');
    elements.candidatePillBtn.classList.add('flex');
    const name = state.profile.name || 'Candidate';
    elements.candidateInitial.innerText = name.charAt(0).toUpperCase();
    elements.candidateNameBadge.innerText = name;
    elements.candidateSkillsCount.innerText = `${(state.profile.skills || []).length} skills`;
  } else {
    elements.candidatePillBtn.classList.add('hidden');
    elements.candidatePillBtn.classList.remove('flex');
  }
}

function openProfileDrawer() {
  if (!state.profile) return;
  const p = state.profile;
  elements.drawerAvatar.innerText = (p.name || 'C').charAt(0).toUpperCase();
  elements.drawerName.innerText = p.name || 'Candidate Profile';
  elements.drawerEmail.innerText = p.email || 'Email not specified';
  elements.drawerExpLevel.innerText = p.experience_level || 'Mid-Level';
  elements.drawerExpYears.innerText = p.experience_years || 'Experienced';
  elements.drawerSummary.innerText = p.summary || 'Extracted via Parser Agent.';

  // Render skills tags
  elements.drawerSkillsTags.innerHTML = (p.skills || []).map(s => 
    `<span class="px-2.5 py-1 rounded-lg bg-slate-800 text-xs font-mono text-brand-300 border border-slate-700">${escapeHtml(s)}</span>`
  ).join('');

  // Render roles
  elements.drawerRoles.innerHTML = (p.preferred_roles || ['Software Engineer']).map(r =>
    `<span class="px-2.5 py-1 rounded-lg bg-indigo-950/60 text-xs text-indigo-300 border border-indigo-800">${escapeHtml(r)}</span>`
  ).join('');

  elements.profileDrawer.classList.remove('hidden');
}

function closeProfileDrawer() {
  elements.profileDrawer.classList.add('hidden');
}

function resetSession() {
  if (confirm('Start a fresh chat session?')) {
    localStorage.removeItem('hidani_session_id');
    localStorage.removeItem('hidani_candidate_profile');
    localStorage.removeItem('hidani_applied_jobs');
    window.location.reload();
  }
}

async function fetchSystemHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      const count = data.database?.total_jobs || 1600;
      elements.jobsCountLabel.innerText = `${count.toLocaleString()} Jobs Ready`;
    }
  } catch (e) {
    console.warn('Backend health check warning:', e.message);
  }
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  const bg = type === 'error' ? 'bg-rose-900 border-rose-700 text-rose-100' : 'bg-slate-900 border-slate-700 text-slate-100';
  toast.className = `px-4 py-2.5 rounded-xl border ${bg} text-xs shadow-xl transition-all duration-300 transform translate-y-2 opacity-0 flex items-center gap-2 pointer-events-auto`;
  toast.innerHTML = `<span>${escapeHtml(message)}</span>`;
  
  elements.toastContainer.appendChild(toast);
  requestAnimationFrame(() => {
    toast.classList.remove('translate-y-2', 'opacity-0');
  });

  setTimeout(() => {
    toast.classList.add('opacity-0', 'translate-y-2');
    setTimeout(() => toast.remove(), 350);
  }, 3500);
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
