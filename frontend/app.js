/**
 * Hidani Careers - Job Application Workspace
 * Handles chat state, resume document parsing, API communication,
 * interactive jobs browsing, applications management, and diagnostics.
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
  appliedJobIds: new Set(JSON.parse(localStorage.getItem('hidani_applied_jobs') || '[]')),
  bookmarkedJobIds: new Set(JSON.parse(localStorage.getItem('hidani_bookmarked_jobs') || '[]')),
  systemHealth: null,
  activeFilter: ''
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

  // Interactive Nav Elements
  navBrowseJobsBtn: document.getElementById('navBrowseJobsBtn'),
  navApplicationsBtn: document.getElementById('navApplicationsBtn'),
  navAppCountBadge: document.getElementById('navAppCountBadge'),
  navProfileBtn: document.getElementById('navProfileBtn'),
  systemStatusBtn: document.getElementById('systemStatusBtn'),
  jobsCountLabel: document.getElementById('jobsCountLabel'),

  // Jobs Modal Elements
  jobsModal: document.getElementById('jobsModal'),
  closeJobsModalBtn: document.getElementById('closeJobsModalBtn'),
  jobsSearchInput: document.getElementById('jobsSearchInput'),
  jobsListContainer: document.getElementById('jobsListContainer'),
  jobsModalSubtitle: document.getElementById('jobsModalSubtitle'),
  jobFilterPills: document.getElementById('jobFilterPills'),

  // Applications Modal Elements
  applicationsModal: document.getElementById('applicationsModal'),
  closeAppsModalBtn: document.getElementById('closeAppsModalBtn'),
  appsListContainer: document.getElementById('appsListContainer'),
  appsModalSubtitle: document.getElementById('appsModalSubtitle'),

  // Diagnostics Modal Elements
  diagnosticsModal: document.getElementById('diagnosticsModal'),
  closeDiagnosticsBtn: document.getElementById('closeDiagnosticsBtn'),
  diagApiStatus: document.getElementById('diagApiStatus'),
  diagJobsCount: document.getElementById('diagJobsCount'),
  diagAppsCount: document.getElementById('diagAppsCount'),
  diagLlmStatus: document.getElementById('diagLlmStatus'),
  diagSessionId: document.getElementById('diagSessionId'),
  copySessionIdBtn: document.getElementById('copySessionIdBtn'),

  // Profile Drawer Elements
  profileDrawer: document.getElementById('profileDrawer'),
  closeDrawerBtn: document.getElementById('closeDrawerBtn'),
  drawerName: document.getElementById('drawerName'),
  drawerEmail: document.getElementById('drawerEmail'),
  drawerExpLevel: document.getElementById('drawerExpLevel'),
  drawerExpYears: document.getElementById('drawerExpYears'),
  drawerSkillsTags: document.getElementById('drawerSkillsTags'),
  drawerSummary: document.getElementById('drawerSummary'),
  drawerRoles: document.getElementById('drawerRoles'),

  // Notification Toast Container
  toastContainer: document.getElementById('toastContainer')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  updateProfileUI();
  fetchSystemHealth();
  updateApplicationsBadge();
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

  elements.dropOverlay.addEventListener('dragleave', (e) => {
    if (e.relatedTarget === null) {
      elements.dropOverlay.classList.add('hidden');
      elements.dropOverlay.classList.remove('flex');
    }
  });
  elements.dropOverlay.addEventListener('dragover', (e) => e.preventDefault());

  window.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.dropOverlay.classList.add('hidden');
    elements.dropOverlay.classList.remove('flex');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  // Interactive Nav Actions
  elements.navBrowseJobsBtn.addEventListener('click', () => openJobsModal());
  elements.closeJobsModalBtn.addEventListener('click', () => closeJobsModal());
  elements.jobsModal.addEventListener('click', (e) => {
    if (e.target === elements.jobsModal) closeJobsModal();
  });

  // Jobs live search
  let searchTimeout = null;
  elements.jobsSearchInput.addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      fetchAndRenderJobsList(e.target.value.trim());
    }, 280);
  });

  // Quick filter pills
  if (elements.jobFilterPills) {
    elements.jobFilterPills.addEventListener('click', (e) => {
      const btn = e.target.closest('.job-filter-btn');
      if (!btn) return;
      
      // Update visual active state
      elements.jobFilterPills.querySelectorAll('.job-filter-btn').forEach(b => {
        b.className = 'job-filter-btn px-2.5 py-1 rounded text-[11px] font-mono bg-neutral-800 text-neutral-300 hover:bg-neutral-700 border border-neutral-700';
      });
      btn.className = 'job-filter-btn px-2.5 py-1 rounded text-[11px] font-mono bg-neutral-100 text-neutral-950 font-semibold';
      
      const filter = btn.getAttribute('data-filter') || '';
      elements.jobsSearchInput.value = filter;
      fetchAndRenderJobsList(filter);
    });
  }

  // Applications Modal
  elements.navApplicationsBtn.addEventListener('click', () => openApplicationsModal());
  elements.closeAppsModalBtn.addEventListener('click', () => closeApplicationsModal());
  elements.applicationsModal.addEventListener('click', (e) => {
    if (e.target === elements.applicationsModal) closeApplicationsModal();
  });

  // Diagnostics Modal
  elements.systemStatusBtn.addEventListener('click', () => openDiagnosticsModal());
  elements.closeDiagnosticsBtn.addEventListener('click', () => closeDiagnosticsModal());
  elements.diagnosticsModal.addEventListener('click', (e) => {
    if (e.target === elements.diagnosticsModal) closeDiagnosticsModal();
  });
  elements.copySessionIdBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(state.sessionId);
    showToast('Session ID copied to clipboard');
  });

  // Profile Drawer
  elements.navProfileBtn.addEventListener('click', () => openProfileDrawer());
  elements.closeDrawerBtn.addEventListener('click', () => closeProfileDrawer());
  elements.profileDrawer.addEventListener('click', (e) => {
    if (e.target === elements.profileDrawer) closeProfileDrawer();
  });

  // Reset Session
  elements.resetChatBtn.addEventListener('click', () => resetSession());

  // Close open modals and drawers on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeAllModals();
    }
  });

  // Global helper for quick prompts
  window.sendPrompt = (promptText) => {
    elements.messageInput.value = promptText;
    handleSendMessage();
  };

  // Global helper for applying
  window.applyToJob = (jobId) => {
    elements.messageInput.value = `apply to ${jobId}`;
    handleSendMessage();
  };

  // Global helper for saving / bookmarking
  window.bookmarkJob = async (jobId, btnEl) => {
    try {
      const isAlready = state.bookmarkedJobIds.has(jobId);
      const endpoint = `${API_BASE}/bookmarks/${encodeURIComponent(jobId)}`;
      const method = isAlready ? 'DELETE' : 'POST';
      const res = await fetch(endpoint, { method });
      const data = await res.json();
      if (!res.ok) {
        showToast(data.detail || 'Failed to update bookmark', 'error');
        return;
      }
      if (isAlready) {
        state.bookmarkedJobIds.delete(jobId);
        showToast('Bookmark removed', 'info');
        if (btnEl) {
          btnEl.innerHTML = 'Save';
          btnEl.className = 'px-3 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 hover:text-white text-xs font-medium border border-neutral-700 transition-colors';
        }
      } else {
        state.bookmarkedJobIds.add(jobId);
        showToast(data.message || 'Job bookmarked!', 'success');
        if (btnEl) {
          btnEl.innerHTML = '★ Saved';
          btnEl.className = 'px-3 py-1.5 rounded bg-amber-950/60 text-amber-300 hover:bg-amber-900/80 text-xs font-medium border border-amber-800/80 transition-colors';
        }
      }
      localStorage.setItem('hidani_bookmarked_jobs', JSON.stringify(Array.from(state.bookmarkedJobIds)));
    } catch (err) {
      console.error('Error bookmarking job:', err);
      showToast('Could not save bookmark', 'error');
    }
  };

  // Global helper to check skill gap
  window.checkSkillGap = (jobId) => {
    elements.messageInput.value = `skills needed for ${jobId}`;
    handleSendMessage();
  };

  // Global helper to clear profile
  window.clearStoredProfile = () => {
    state.profile = null;
    localStorage.removeItem('hidani_candidate_profile');
    updateProfileUI();
    closeProfileDrawer();
    showToast('Candidate profile cleared');
  };
}

/**
 * Send user message to Backend API
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

    // Refresh application counts if applied
    updateApplicationsBadge();

  } catch (error) {
    removeTypingIndicator(typingIndicatorId);
    appendSystemAlert(`Error contacting server: ${error.message}. Please verify the backend is running.`);
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
  const uploadActivityId = appendActivityMessage(`Processing document: ${file.name}...`);
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

    // Format fields cleanly
    const skillsFormatted = (profile.skills || []).join(', ') || 'Software Engineering, Python, REST APIs';
    const eduFormatted = (profile.education || []).join(' | ') || 'Computer Science / Engineering Degree';
    const projFormatted = (profile.projects || []).map(p => `  • ${p}`).join('\n') || '  • Engineering and software development projects';
    const contactInfo = profile.contact || [profile.email, profile.phone].filter(Boolean).join(' | ') || 'Provided in document';

    const parseSummaryMarkdown = (
      `### Resume Parsed Successfully\n\n` +
      `| Field | Details |\n` +
      `| :--- | :--- |\n` +
      `| **Candidate** | **${profile.name || 'Candidate'}** |\n` +
      `| **Contact** | ${contactInfo} |\n` +
      `| **Designation** | **${profile.designation || 'Software Engineer'}** |\n` +
      `| **Skills** | ${skillsFormatted} |\n` +
      `| **Experience** | ${profile.experience || profile.experience_years || '1-3 years'} (${profile.experience_level || 'Mid-Level'}) |\n` +
      `| **Education** | ${eduFormatted} |\n\n` +
      `**Projects:**\n${projFormatted}\n\n` +
      `*Searching database for positions matching this candidate profile...*`
    );

    appendAssistantMessage(parseSummaryMarkdown);
    showToast(`Parsed resume for ${profile.name || 'Candidate'}`, 'success');

    // Auto-trigger matching jobs
    const recActivityId = appendActivityMessage('Matching candidate profile against database listings...');
    scrollToBottom();

    try {
      const recRes = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: `Recommend jobs matching designation (${profile.designation || 'Software Engineer'}), skills (${(profile.skills || []).slice(0, 5).join(', ')}), and experience level (${profile.experience_level || 'Mid-Level'})`,
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
    appendSystemAlert(`Document Processing Failed: ${error.message}`);
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
    <div class="max-w-[85%] sm:max-w-[75%] bg-neutral-800 border border-neutral-700/70 text-neutral-100 rounded-xl px-4 py-2.5 shadow-sm text-sm leading-relaxed">
      <p class="whitespace-pre-wrap">${escapeHtml(text)}</p>
    </div>
  `;
  elements.messagesContainer.appendChild(msgEl);
}

function appendAssistantMessage(markdownContent, actions = []) {
  const msgEl = document.createElement('div');
  msgEl.className = 'message-appear';

  const parsedHtml = marked.parse(markdownContent || '');

  msgEl.innerHTML = `
    <div class="bg-neutral-900 border border-neutral-800 rounded-xl p-4 sm:p-5 shadow-sm text-neutral-200 text-sm">
      <div class="text-[10px] font-mono uppercase tracking-wider text-neutral-400 mb-2 font-semibold">Assistant</div>
      <div class="prose-chat">${parsedHtml}</div>
      <div class="actions-slot mt-3 space-y-3"></div>
    </div>
  `;

  elements.messagesContainer.appendChild(msgEl);

  const actionsSlot = msgEl.querySelector('.actions-slot');
  if (actions && actions.length > 0) {
    actions.forEach(action => {
      if (action.type === 'RECOMMEND_JOBS' && Array.isArray(action.data)) {
        renderInteractiveJobCards(actionsSlot, action.data);
      } else if (action.type === 'APPLICATION_SUBMITTED' && action.data) {
        renderApplicationBadge(actionsSlot, action.data);
      } else if (action.type === 'BOOKMARK_SAVED' && action.data) {
        renderBookmarkBadge(actionsSlot, action.data);
      } else if (action.type === 'SKILL_GAP_ANALYSIS' && action.data) {
        renderSkillGapBadge(actionsSlot, action.data);
      } else if (action.type === 'REMINDER_SET' && action.data) {
        renderReminderBadge(actionsSlot, action.data);
      }
    });
  }
}

/**
 * Render Job Cards inside Assistant message
 */
function renderInteractiveJobCards(container, jobs) {
  if (!jobs || jobs.length === 0) return;

  const cardList = document.createElement('div');
  cardList.className = 'space-y-3 pt-2';

  jobs.forEach((job, idx) => {
    const isApplied = state.appliedJobIds.has(job.job_id);
    const isBookmarked = state.bookmarkedJobIds.has(job.job_id);
    const score = job.match_score || 85;

    const card = document.createElement('div');
    card.className = 'p-4 rounded-lg bg-neutral-950 border border-neutral-800 hover:border-neutral-700 transition-colors shadow-sm';
    
    const skillPills = (job.skills || []).slice(0, 5).map(s => 
      `<span class="px-2 py-0.5 rounded bg-neutral-800 text-[11px] text-neutral-300 font-mono border border-neutral-700/60">${escapeHtml(s)}</span>`
    ).join(' ');

    card.innerHTML = `
      <div class="flex items-start justify-between gap-3 mb-2">
        <div>
          <div class="flex items-center gap-2">
            <span class="text-xs font-mono text-neutral-400">${idx + 1}.</span>
            <h4 class="text-sm font-semibold text-white">${escapeHtml(job.title)}</h4>
          </div>
          <p class="text-xs text-neutral-400 mt-0.5">${escapeHtml(job.company)} • <span class="text-neutral-300">${escapeHtml(job.location)}</span></p>
        </div>
        <div class="text-right flex-shrink-0">
          <span class="inline-block px-2 py-0.5 rounded bg-neutral-800 border border-neutral-700 text-neutral-200 text-xs font-mono font-medium">
            ${score}% Match
          </span>
        </div>
      </div>

      <div class="text-xs text-neutral-300 bg-neutral-900 rounded p-2.5 mb-3 border border-neutral-800 leading-relaxed">
        <span class="text-neutral-400 font-medium block text-[11px] uppercase tracking-wider mb-0.5">Match Assessment:</span>
        ${escapeHtml(job.match_reason || 'Strong alignment with candidate skills and experience level.')}
      </div>

      <div class="flex flex-wrap items-center gap-1.5 mb-3">
        <span class="px-2 py-0.5 rounded bg-neutral-800 text-neutral-300 border border-neutral-700 text-[10px] font-mono uppercase">${escapeHtml(job.experience || 'Mid-Level')}</span>
        ${skillPills}
      </div>

      <div class="flex items-center justify-between pt-2.5 border-t border-neutral-800">
        <span class="font-mono text-[10px] text-neutral-500">ID: ${escapeHtml(job.job_id)}</span>
        <div class="flex items-center gap-2">
          ${job.source_link ? `
            <a href="${escapeHtml(job.source_link)}" target="_blank" rel="noopener noreferrer" class="px-3 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 hover:text-white text-xs font-medium border border-neutral-700 transition-colors">
              View Listing
            </a>
          ` : ''}
          <button
            onclick="window.bookmarkJob('${escapeHtml(job.job_id)}', this)"
            title="${isBookmarked ? 'Remove bookmark' : 'Save job'}"
            class="px-3 py-1.5 rounded text-xs font-medium border transition-colors ${
              isBookmarked
                ? 'bg-amber-950/60 text-amber-300 hover:bg-amber-900/80 border-amber-800/80'
                : 'bg-neutral-800 hover:bg-neutral-700 text-neutral-300 hover:text-white border-neutral-700'
            }"
          >
            ${isBookmarked ? '★ Saved' : 'Save'}
          </button>
          <button
            onclick="window.applyToJob('${escapeHtml(job.job_id)}')"
            ${isApplied ? 'disabled' : ''}
            class="px-3.5 py-1.5 rounded text-xs font-semibold transition-colors ${
              isApplied
                ? 'bg-neutral-800 text-neutral-500 cursor-not-allowed border border-neutral-700'
                : 'bg-neutral-100 hover:bg-white text-neutral-950'
            }"
          >
            ${isApplied ? 'Applied' : 'Apply'}
          </button>
        </div>
      </div>
    `;

    cardList.appendChild(card);
  });

  container.appendChild(cardList);
}

/**
 * Render bookmark confirmation widget
 */
function renderBookmarkBadge(container, data) {
  if (!data) return;
  const badge = document.createElement('div');
  const isSuccess = data.success !== false;
  const bm = data.bookmark || {};
  if (isSuccess && (data.job_id || bm.job_id)) {
    const jId = data.job_id || bm.job_id;
    state.bookmarkedJobIds.add(jId);
    localStorage.setItem('hidani_bookmarked_jobs', JSON.stringify(Array.from(state.bookmarkedJobIds)));
  }

  badge.className = `p-3 rounded-lg border text-xs flex items-center justify-between gap-3 ${
    isSuccess
      ? 'bg-neutral-900 border-amber-800/60 text-amber-200'
      : 'bg-neutral-900 border-neutral-700 text-neutral-300'
  }`;

  badge.innerHTML = `
    <div class="flex items-center gap-2">
      <span class="text-amber-400 text-sm">★</span>
      <div>
        <span class="font-semibold text-white">${escapeHtml(data.message || 'Job bookmark updated')}</span>
        ${bm.job_title ? `<div class="text-[11px] text-neutral-400 mt-0.5">${escapeHtml(bm.job_title)} • ${escapeHtml(bm.company || '')}</div>` : ''}
      </div>
    </div>
    <span class="font-mono text-[10px] px-2 py-0.5 rounded bg-neutral-950 border border-neutral-800 text-neutral-400">
      ${escapeHtml(data.job_id || bm.job_id || 'Bookmark')}
    </span>
  `;
  container.appendChild(badge);
}

/**
 * Render skill gap analysis widget
 */
function renderSkillGapBadge(container, analysis) {
  if (!analysis) return;
  const badge = document.createElement('div');
  badge.className = 'p-3.5 rounded-lg border border-neutral-800 bg-neutral-950 text-xs shadow-sm';

  const matchPills = (analysis.matching_skills || []).map(s => 
    `<span class="px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-300 font-mono text-[10px] border border-emerald-800/60">${escapeHtml(s)}</span>`
  ).join(' ') || '<span class="text-neutral-500 italic">None</span>';

  const missingPills = (analysis.missing_skills || []).map(s => 
    `<span class="px-2 py-0.5 rounded bg-rose-950/60 text-rose-300 font-mono text-[10px] border border-rose-800/60">${escapeHtml(s)}</span>`
  ).join(' ') || '<span class="text-emerald-400">All required skills met!</span>';

  badge.innerHTML = `
    <div class="flex items-center justify-between mb-2">
      <div class="font-semibold text-white">${escapeHtml(analysis.job_title)} • ${escapeHtml(analysis.company)}</div>
      <span class="px-2 py-0.5 rounded bg-neutral-800 text-neutral-200 font-mono text-[11px]">${analysis.match_score}% Score</span>
    </div>
    <div class="space-y-1.5 mb-2.5">
      <div><span class="text-neutral-400 text-[11px]">Matching Skills:</span> ${matchPills}</div>
      <div><span class="text-neutral-400 text-[11px]">Missing / Learn:</span> ${missingPills}</div>
    </div>
    <div class="p-2 rounded bg-neutral-900 border border-neutral-800 text-neutral-300 text-[11px] leading-relaxed">
      ${escapeHtml(analysis.recommendation || '')}
    </div>
  `;
  container.appendChild(badge);
}

/**
 * Render follow-up reminder confirmation widget
 */
function renderReminderBadge(container, data) {
  if (!data) return;
  const badge = document.createElement('div');
  const isSuccess = data.success !== false;
  badge.className = `p-3 rounded-lg border text-xs flex items-center justify-between gap-3 ${
    isSuccess
      ? 'bg-neutral-900 border-indigo-800/60 text-indigo-200'
      : 'bg-neutral-900 border-neutral-700 text-neutral-300'
  }`;

  badge.innerHTML = `
    <div class="flex items-center gap-2">
      <span class="text-indigo-400">⏱</span>
      <div>
        <span class="font-semibold text-white">${escapeHtml(data.message || 'Follow-up reminder updated')}</span>
        ${data.application && data.application.reminder_date ? `<div class="text-[11px] text-neutral-400 mt-0.5">Target date: ${escapeHtml(data.application.reminder_date)}</div>` : ''}
      </div>
    </div>
    <span class="font-mono text-[10px] px-2 py-0.5 rounded bg-neutral-950 border border-neutral-800 text-neutral-400">
      ${escapeHtml(data.job_id || 'Reminder')}
    </span>
  `;
  container.appendChild(badge);
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
    updateApplicationsBadge();
  }

  badge.className = `p-3 rounded-lg border ${
    isSuccess
      ? 'bg-neutral-900 border-neutral-700 text-neutral-200'
      : 'bg-neutral-900 border-neutral-700 text-neutral-300'
  } text-xs`;

  badge.innerHTML = `
    <div class="flex items-center gap-2 mb-1">
      <span class="w-1.5 h-1.5 rounded-full ${isSuccess ? 'bg-emerald-400' : 'bg-neutral-400'}"></span>
      <span class="font-semibold text-white">${isSuccess ? 'Application Recorded' : 'Application Notice'}</span>
    </div>
    <p class="text-neutral-400">${escapeHtml(appData.message)}</p>
  `;
  container.appendChild(badge);
}

function appendActivityMessage(text) {
  const id = `act_${Date.now()}`;
  const el = document.createElement('div');
  el.id = id;
  el.className = 'flex items-center gap-2 px-3 py-2 rounded bg-neutral-900 border border-neutral-800 text-xs text-neutral-400 message-appear w-fit';
  el.innerHTML = `
    <span class="w-1.5 h-1.5 rounded-full bg-neutral-400 animate-pulse"></span>
    <span>${escapeHtml(text)}</span>
  `;
  elements.messagesContainer.appendChild(el);
  return id;
}

function appendSystemAlert(text) {
  const el = document.createElement('div');
  el.className = 'p-3 rounded-lg bg-neutral-900 border border-neutral-700 text-xs text-neutral-300 message-appear';
  el.innerHTML = `<span class="font-semibold text-white">Notice:</span> ${escapeHtml(text)}`;
  elements.messagesContainer.appendChild(el);
}

/**
 * Typing Indicator with subtle pulsing dots
 */
function showTypingIndicator() {
  const id = `typing_${Date.now()}`;
  const el = document.createElement('div');
  el.id = id;
  el.className = 'message-appear';
  el.innerHTML = `
    <div class="bg-neutral-900 border border-neutral-800 rounded-xl px-4 py-3 shadow-sm inline-flex items-center gap-1.5">
      <span class="w-1.5 h-1.5 rounded-full bg-neutral-500 animate-pulse" style="animation-delay: 0ms"></span>
      <span class="w-1.5 h-1.5 rounded-full bg-neutral-500 animate-pulse" style="animation-delay: 200ms"></span>
      <span class="w-1.5 h-1.5 rounded-full bg-neutral-500 animate-pulse" style="animation-delay: 400ms"></span>
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
 * Interactive Jobs Modal Handling
 */
function openJobsModal() {
  elements.jobsModal.classList.remove('hidden');
  fetchAndRenderJobsList(elements.jobsSearchInput.value.trim());
  elements.jobsSearchInput.focus();
}

function closeJobsModal() {
  elements.jobsModal.classList.add('hidden');
}

async function fetchAndRenderJobsList(keyword = '') {
  elements.jobsListContainer.innerHTML = `
    <div class="py-12 text-center text-xs text-neutral-400 font-mono">
      <span class="inline-block w-2 h-2 rounded-full bg-neutral-400 animate-ping mr-2"></span>
      Loading positions...
    </div>
  `;

  try {
    const url = keyword
      ? `${API_BASE}/jobs?query=${encodeURIComponent(keyword)}&limit=30`
      : `${API_BASE}/jobs?limit=30`;
    
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const jobs = data.jobs || [];

    elements.jobsModalSubtitle.innerText = `${data.total.toLocaleString()} total positions available`;

    if (jobs.length === 0) {
      elements.jobsListContainer.innerHTML = `
        <div class="py-12 text-center text-xs text-neutral-400">
          No positions found matching "${escapeHtml(keyword)}". Try another keyword.
        </div>
      `;
      return;
    }

    elements.jobsListContainer.innerHTML = '';
    jobs.forEach(job => {
      const isApplied = state.appliedJobIds.has(job.job_id);
      const isBookmarked = state.bookmarkedJobIds.has(job.job_id);
      const row = document.createElement('div');
      row.className = 'p-3.5 rounded-lg bg-neutral-950 border border-neutral-800 hover:border-neutral-700 transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-3';
      
      const skillPills = (job.skills || []).slice(0, 3).map(s => 
        `<span class="px-2 py-0.5 rounded bg-neutral-800 text-[10px] text-neutral-300 font-mono">${escapeHtml(s)}</span>`
      ).join(' ');

      row.innerHTML = `
        <div class="flex-1">
          <div class="flex items-center gap-2">
            <h4 class="text-xs sm:text-sm font-semibold text-white">${escapeHtml(job.title)}</h4>
            <span class="text-[10px] font-mono px-1.5 py-0.2 rounded bg-neutral-800 text-neutral-400 border border-neutral-700">${escapeHtml(job.experience || 'All Levels')}</span>
          </div>
          <p class="text-xs text-neutral-400 mt-0.5">${escapeHtml(job.company)} • <span class="text-neutral-300">${escapeHtml(job.location)}</span></p>
          <div class="flex flex-wrap gap-1 mt-2">${skillPills}</div>
        </div>
        <div class="flex items-center gap-2 flex-shrink-0">
          <button
            onclick="discussJob('${escapeHtml(job.job_id)}', '${escapeHtml(job.title)}')"
            class="px-2.5 py-1.5 rounded text-xs text-neutral-300 hover:text-white bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 transition-colors"
          >
            Ask AI
          </button>
          <button
            onclick="window.bookmarkJob('${escapeHtml(job.job_id)}', this)"
            title="${isBookmarked ? 'Remove bookmark' : 'Save job'}"
            class="px-2.5 py-1.5 rounded text-xs font-medium border transition-colors ${
              isBookmarked
                ? 'bg-amber-950/60 text-amber-300 hover:bg-amber-900/80 border-amber-800/80'
                : 'bg-neutral-800 hover:bg-neutral-700 text-neutral-300 hover:text-white border-neutral-700'
            }"
          >
            ${isBookmarked ? '★ Saved' : 'Save'}
          </button>
          <button
            onclick="quickApplyFromModal('${escapeHtml(job.job_id)}')"
            ${isApplied ? 'disabled' : ''}
            class="px-3 py-1.5 rounded text-xs font-semibold transition-colors ${
              isApplied
                ? 'bg-neutral-800 text-neutral-500 cursor-not-allowed border border-neutral-700'
                : 'bg-neutral-100 hover:bg-white text-neutral-950'
            }"
          >
            ${isApplied ? 'Applied' : 'Apply'}
          </button>
        </div>
      `;
      elements.jobsListContainer.appendChild(row);
    });

  } catch (err) {
    elements.jobsListContainer.innerHTML = `
      <div class="p-6 text-center text-xs text-rose-400">
        Failed to fetch jobs: ${escapeHtml(err.message)}
      </div>
    `;
  }
}

window.discussJob = (jobId, title) => {
  closeJobsModal();
  elements.messageInput.value = `Tell me about job ${jobId} (${title})`;
  handleSendMessage();
};

window.quickApplyFromModal = (jobId) => {
  closeJobsModal();
  window.applyToJob(jobId);
};

/**
 * Interactive Applications Modal
 */
function openApplicationsModal() {
  elements.applicationsModal.classList.remove('hidden');
  fetchAndRenderApplicationsList();
}

function closeApplicationsModal() {
  elements.applicationsModal.classList.add('hidden');
}

async function fetchAndRenderApplicationsList() {
  elements.appsListContainer.innerHTML = `
    <div class="py-12 text-center text-xs text-neutral-400 font-mono">
      Loading submitted applications...
    </div>
  `;

  try {
    const res = await fetch(`${API_BASE}/applications`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const apps = await res.json();

    elements.appsModalSubtitle.innerText = `${apps.length} total applications submitted`;
    updateApplicationsBadge(apps.length);

    if (apps.length === 0) {
      elements.appsListContainer.innerHTML = `
        <div class="py-12 text-center space-y-2">
          <p class="text-xs text-neutral-400">No applications have been recorded yet.</p>
          <button onclick="closeApplicationsModal(); openJobsModal();" class="px-3 py-1.5 rounded bg-neutral-100 hover:bg-white text-neutral-950 text-xs font-semibold transition-colors">
            Browse Openings to Apply
          </button>
        </div>
      `;
      return;
    }

    elements.appsListContainer.innerHTML = '';
    const statuses = ['Applied', 'Under Review', 'Interview', 'Offer', 'Rejected'];
    apps.forEach((app, idx) => {
      const card = document.createElement('div');
      card.className = 'p-3.5 rounded-lg bg-neutral-950 border border-neutral-800 flex items-start justify-between gap-3';
      const currentStatus = app.status || 'Applied';
      const statusOptions = statuses.map(s =>
        `<option value="${s}" ${s === currentStatus ? 'selected' : ''}>${s}</option>`
      ).join('');
      card.innerHTML = `
        <div>
          <div class="flex items-center gap-2">
            <span class="text-xs font-mono text-neutral-400">#${idx + 1}</span>
            <h4 class="text-xs sm:text-sm font-semibold text-white">${escapeHtml(app.job_title || 'Software Role')}</h4>
          </div>
          <p class="text-xs text-neutral-400 mt-0.5">${escapeHtml(app.company || 'Tech Company')} • <span class="text-neutral-300">${escapeHtml(app.location || 'Remote')}</span></p>
          <p class="text-[11px] font-mono text-neutral-500 mt-1">Submitted on ${escapeHtml(app.applied_date || 'Recent')}</p>
        </div>
        <div class="text-right flex-shrink-0 flex flex-col items-end gap-2">
          <select
            onchange="updateApplicationStatusFromModal('${escapeHtml(app.job_id)}', this.value)"
            aria-label="Update application status"
            class="px-2 py-0.5 rounded text-[11px] font-mono font-medium bg-neutral-800 text-neutral-200 border border-neutral-700 focus:outline-none focus:border-neutral-500 cursor-pointer"
          >
            ${statusOptions}
          </select>
          <button
            onclick="withdrawApplicationFromModal('${escapeHtml(app.job_id)}')"
            title="Withdraw this application"
            class="px-2 py-0.5 rounded text-[10px] text-neutral-400 hover:text-rose-400 hover:bg-neutral-900 border border-neutral-800 hover:border-neutral-700 transition-colors"
          >
            Withdraw
          </button>
        </div>
      `;
      elements.appsListContainer.appendChild(card);
    });

  } catch (err) {
    elements.appsListContainer.innerHTML = `
      <div class="p-6 text-center text-xs text-rose-400">
        Failed to fetch applications: ${escapeHtml(err.message)}
      </div>
    `;
  }
}

async function updateApplicationsBadge(knownCount = null) {
  try {
    let count = knownCount;
    if (count === null) {
      const res = await fetch(`${API_BASE}/applications`);
      if (res.ok) {
        const apps = await res.json();
        count = apps.length;
        // Rebuild applied set so withdrawn/removed jobs are cleared from localStorage
        state.appliedJobIds = new Set(apps.map(a => a.job_id).filter(Boolean));
        localStorage.setItem('hidani_applied_jobs', JSON.stringify(Array.from(state.appliedJobIds)));
      }
    }

    if (count !== null && count > 0) {
      elements.navAppCountBadge.innerText = count;
      elements.navAppCountBadge.classList.remove('hidden');
    } else {
      elements.navAppCountBadge.classList.add('hidden');
    }
  } catch (e) {
    // Silently ignore if offline
  }
}

/**
 * Diagnostics Modal
 */
function openDiagnosticsModal() {
  elements.diagSessionId.innerText = state.sessionId;
  elements.diagAppsCount.innerText = `${state.appliedJobIds.size} tracked in session`;
  elements.diagnosticsModal.classList.remove('hidden');
  fetchSystemHealth();
}

function closeDiagnosticsModal() {
  elements.diagnosticsModal.classList.add('hidden');
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
    elements.navProfileBtn.classList.remove('hidden');
  } else {
    elements.navProfileBtn.classList.add('hidden');
  }
}

function openProfileDrawer() {
  if (!state.profile) {
    showToast('No resume uploaded yet');
    return;
  }
  const p = state.profile;
  elements.drawerName.innerText = p.name || 'Candidate Profile';
  elements.drawerEmail.innerText = p.email || 'Email not specified';
  elements.drawerExpLevel.innerText = p.experience_level || 'Mid-Level';
  elements.drawerExpYears.innerText = p.experience_years || 'Experienced';
  elements.drawerSummary.innerText = p.summary || 'Extracted from resume document.';

  // Render skills tags
  elements.drawerSkillsTags.innerHTML = (p.skills || []).map(s => 
    `<span class="px-2 py-0.5 rounded bg-neutral-800 text-xs font-mono text-neutral-300 border border-neutral-700">${escapeHtml(s)}</span>`
  ).join('');

  // Render roles
  elements.drawerRoles.innerHTML = (p.preferred_roles || ['Software Engineer']).map(r =>
    `<span class="px-2 py-0.5 rounded bg-neutral-800 text-xs text-neutral-300 border border-neutral-700">${escapeHtml(r)}</span>`
  ).join('');

  elements.profileDrawer.classList.remove('hidden');
}

function closeProfileDrawer() {
  elements.profileDrawer.classList.add('hidden');
}

function closeAllModals() {
  closeJobsModal();
  closeApplicationsModal();
  closeDiagnosticsModal();
  closeProfileDrawer();
}

window.withdrawApplicationFromModal = async (jobId) => {
  if (!confirm(`Are you sure you want to withdraw your application for Job ID ${jobId}?`)) return;
  try {
    const res = await fetch(`${API_BASE}/applications/${encodeURIComponent(jobId)}`, {
      method: 'DELETE'
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Withdrawal failed (${res.status})`);
    }
    state.appliedJobIds.delete(jobId);
    localStorage.setItem('hidani_applied_jobs', JSON.stringify(Array.from(state.appliedJobIds)));
    showToast(`Withdrew application for ${jobId}`);
    fetchAndRenderApplicationsList();
    updateApplicationsBadge();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
};

window.updateApplicationStatusFromModal = async (jobId, newStatus) => {
  try {
    const res = await fetch(`${API_BASE}/applications/${encodeURIComponent(jobId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: newStatus })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Status update failed (${res.status})`);
    }
    showToast(`Updated ${jobId} to ${newStatus}`, 'success');
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
    fetchAndRenderApplicationsList();
  }
};

function resetSession() {
  if (confirm('Start a fresh workspace session?')) {
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
      state.systemHealth = data;
      const count = data.database?.total_jobs || 1600;
      elements.jobsCountLabel.innerText = `${count.toLocaleString()} Jobs`;
      
      if (elements.diagJobsCount) elements.diagJobsCount.innerText = `${count.toLocaleString()} Ready`;
      if (elements.diagLlmStatus) elements.diagLlmStatus.innerText = data.openai_configured ? 'Active (Live LLM)' : 'Autonomous Fallback';
      if (elements.diagApiStatus) elements.diagApiStatus.innerText = 'Online (200 OK)';
    }
  } catch (e) {
    elements.jobsCountLabel.innerText = 'Offline';
    if (elements.diagApiStatus) elements.diagApiStatus.innerText = 'Offline / Connecting...';
  }
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  const styleByType = {
    success: 'border-emerald-700/80 text-emerald-200',
    error: 'border-rose-700/80 text-rose-200',
    info: 'border-neutral-700 text-neutral-200'
  };
  const dotByType = {
    success: 'bg-emerald-400',
    error: 'bg-rose-400',
    info: 'bg-neutral-400'
  };
  const borderClass = styleByType[type] || styleByType.info;
  const dotClass = dotByType[type] || dotByType.info;

  toast.className = `px-3.5 py-2 rounded border ${borderClass} bg-neutral-900 text-xs shadow-lg transition-all duration-200 transform translate-y-2 opacity-0 pointer-events-auto flex items-center gap-2`;
  toast.innerHTML = `<span class="w-1.5 h-1.5 rounded-full ${dotClass}"></span><span>${escapeHtml(message)}</span>`;
  
  elements.toastContainer.appendChild(toast);
  requestAnimationFrame(() => {
    toast.classList.remove('translate-y-2', 'opacity-0');
  });

  setTimeout(() => {
    toast.classList.add('opacity-0', 'translate-y-2');
    setTimeout(() => toast.remove(), 250);
  }, 3000);
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
