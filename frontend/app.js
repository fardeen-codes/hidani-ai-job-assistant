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
  activeFilter: '',
  minSalary: 0,
  appStatusFilter: 'ALL'
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
  navStreakBadge: document.getElementById('navStreakBadge'),
  navStreakCountText: document.getElementById('navStreakCountText'),
  navDailyGoalText: document.getElementById('navDailyGoalText'),
  navProfileBtn: document.getElementById('navProfileBtn'),
  systemStatusBtn: document.getElementById('systemStatusBtn'),
  jobsCountLabel: document.getElementById('jobsCountLabel'),

  // Jobs Modal Elements
  jobsModal: document.getElementById('jobsModal'),
  closeJobsModalBtn: document.getElementById('closeJobsModalBtn'),
  exportJobsCsvBtn: document.getElementById('exportJobsCsvBtn'),
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

  // Shortcuts Modal Elements
  shortcutsBtn: document.getElementById('shortcutsBtn'),
  shortcutsModal: document.getElementById('shortcutsModal'),
  closeShortcutsBtn: document.getElementById('closeShortcutsBtn'),

  // Profile Drawer Elements
  profileDrawer: document.getElementById('profileDrawer'),
  closeDrawerBtn: document.getElementById('closeDrawerBtn'),
  drawerName: document.getElementById('drawerName'),
  drawerEmail: document.getElementById('drawerEmail'),
  drawerExpLevel: document.getElementById('drawerExpLevel'),
  drawerExpYears: document.getElementById('drawerExpYears'),
  drawerSkillsTags: document.getElementById('drawerSkillsTags'),
  drawerLanguagesTags: document.getElementById('drawerLanguagesTags'),
  drawerSummary: document.getElementById('drawerSummary'),
  drawerRoles: document.getElementById('drawerRoles'),
  drawerLinksList: document.getElementById('drawerLinksList'),
  drawerCompletenessBadge: document.getElementById('drawerCompletenessBadge'),
  drawerCompletenessBar: document.getElementById('drawerCompletenessBar'),
  drawerCompletenessHint: document.getElementById('drawerCompletenessHint'),
  drawerTeamSizeSelect: document.getElementById('drawerTeamSizeSelect'),
  drawerCompanyStageSelect: document.getElementById('drawerCompanyStageSelect'),
  saveTeamPrefsBtn: document.getElementById('saveTeamPrefsBtn'),
  drawerNoticeBadge: document.getElementById('drawerNoticeBadge'),
  drawerStartDate: document.getElementById('drawerStartDate'),

  // Notification Toast Container
  toastContainer: document.getElementById('toastContainer')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  updateProfileUI();
  fetchSystemHealth();
  updateApplicationsBadge();
  fetchAndUpdateStreakMetrics();
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

  // Auto-resize input & char counter
  const messageCharCount = document.getElementById('messageCharCount');
  const clearDraftBtn = document.getElementById('clearDraftBtn');

  elements.messageInput.addEventListener('input', () => {
    elements.messageInput.style.height = 'auto';
    elements.messageInput.style.height = Math.min(elements.messageInput.scrollHeight, 120) + 'px';
    const len = elements.messageInput.value.length;
    if (messageCharCount) {
      messageCharCount.textContent = `${len.toLocaleString()} chars`;
    }
    if (clearDraftBtn) {
      clearDraftBtn.classList.toggle('hidden', len === 0);
    }
  });

  if (clearDraftBtn) {
    clearDraftBtn.addEventListener('click', () => {
      elements.messageInput.value = '';
      elements.messageInput.style.height = 'auto';
      if (messageCharCount) messageCharCount.textContent = '0 chars';
      clearDraftBtn.classList.add('hidden');
      elements.messageInput.focus();
    });
  }

  // Quick prompt chips
  const chipsContainer = document.getElementById('chatQuickPromptChips');
  if (chipsContainer) {
    chipsContainer.addEventListener('click', (e) => {
      const chip = e.target.closest('.quick-chip');
      if (!chip) return;
      if (chip.id === 'pitchQuickBtn') {
        e.preventDefault();
        window.generateElevatorPitch();
        return;
      }
      const prompt = chip.getAttribute('data-prompt');
      if (prompt) {
        elements.messageInput.value = prompt;
        elements.messageInput.focus();
        elements.messageInput.dispatchEvent(new Event('input'));
      }
    });
  }

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
  if (elements.exportJobsCsvBtn) {
    elements.exportJobsCsvBtn.addEventListener('click', () => {
      const q = elements.jobsSearchInput.value.trim();
      const exportUrl = q
        ? `${API_BASE}/jobs/export/csv?query=${encodeURIComponent(q)}`
        : `${API_BASE}/jobs/export/csv`;
      showToast('Exporting jobs to CSV...', 'info');
      const a = document.createElement('a');
      a.href = exportUrl;
      a.download = 'jobs_export.csv';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    });
  }
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

  // Quick salary filter pills
  const salaryPills = document.getElementById('salaryFilterPills');
  if (salaryPills) {
    salaryPills.addEventListener('click', (e) => {
      const btn = e.target.closest('.salary-filter-btn');
      if (!btn) return;
      salaryPills.querySelectorAll('.salary-filter-btn').forEach(b => {
        b.className = 'salary-filter-btn px-2 py-0.5 rounded text-[10px] font-mono bg-neutral-800 text-neutral-300 hover:bg-neutral-700 border border-neutral-700';
      });
      btn.className = 'salary-filter-btn px-2 py-0.5 rounded text-[10px] font-mono bg-neutral-100 text-neutral-950 font-semibold';
      state.minSalary = parseInt(btn.getAttribute('data-min-salary'), 10) || 0;
      fetchAndRenderJobsList(elements.jobsSearchInput.value.trim());
    });
  }

  // Applications Modal
  elements.navApplicationsBtn.addEventListener('click', () => openApplicationsModal());
  elements.closeAppsModalBtn.addEventListener('click', () => closeApplicationsModal());
  elements.applicationsModal.addEventListener('click', (e) => {
    if (e.target === elements.applicationsModal) closeApplicationsModal();
  });

  const exportPortfolioBtn = document.getElementById('exportPortfolioJsonBtn');
  if (exportPortfolioBtn) {
    exportPortfolioBtn.addEventListener('click', async () => {
      try {
        const res = await fetch(`${API_BASE}/applications/export/portfolio`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `application_portfolio_${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        showToast('Application portfolio exported as JSON', 'success');
      } catch (err) {
        showToast(`Portfolio export failed: ${err.message}`, 'error');
      }
    });
  }

  const appStatusFilterBar = document.getElementById('appsStatusFilterBar');
  if (appStatusFilterBar) {
    appStatusFilterBar.addEventListener('click', (e) => {
      const btn = e.target.closest('.app-status-filter-pill');
      if (!btn) return;
      appStatusFilterBar.querySelectorAll('.app-status-filter-pill').forEach(b => {
        b.className = 'app-status-filter-pill px-2.5 py-1 rounded text-xs font-mono transition-colors bg-neutral-800 hover:bg-neutral-700 text-neutral-300';
      });
      btn.className = 'app-status-filter-pill px-2.5 py-1 rounded text-xs font-mono transition-colors bg-neutral-100 text-neutral-950 font-semibold';
      state.appStatusFilter = btn.getAttribute('data-status') || 'ALL';
      fetchAndRenderApplicationsList();
    });
  }

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

  const compactDbBtn = document.getElementById('compactDbBtn');
  if (compactDbBtn) {
    compactDbBtn.addEventListener('click', async () => {
      compactDbBtn.disabled = true;
      compactDbBtn.textContent = 'Compacting...';
      try {
        const res = await fetch(`${API_BASE}/db/compact`, { method: 'POST' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        showToast(data.message || 'Database compacted successfully', 'success');
        openDiagnosticsModal();
      } catch (err) {
        showToast(`Compaction failed: ${err.message}`, 'error');
      } finally {
        compactDbBtn.disabled = false;
        compactDbBtn.textContent = 'Compact';
      }
    });
  }

  // Profile Drawer
  elements.navProfileBtn.addEventListener('click', () => openProfileDrawer());
  elements.closeDrawerBtn.addEventListener('click', () => closeProfileDrawer());
  elements.profileDrawer.addEventListener('click', (e) => {
    if (e.target === elements.profileDrawer) closeProfileDrawer();
  });
  if (elements.saveTeamPrefsBtn) {
    elements.saveTeamPrefsBtn.addEventListener('click', async () => {
      try {
        const teamSize = elements.drawerTeamSizeSelect ? elements.drawerTeamSizeSelect.value : '10-50';
        const companyStage = elements.drawerCompanyStageSelect ? elements.drawerCompanyStageSelect.value : 'Growth';
        elements.saveTeamPrefsBtn.disabled = true;
        elements.saveTeamPrefsBtn.textContent = 'Saving...';
        const res = await fetch(`${API_BASE}/profile/team-preferences`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            preferred_team_size: teamSize,
            preferred_company_stage: companyStage
          })
        });
        if (res.ok) {
          showToast('Team & stage preferences saved!', 'success');
        } else {
          showToast('Failed to save team preferences.', 'error');
        }
      } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
      } finally {
        elements.saveTeamPrefsBtn.disabled = false;
        elements.saveTeamPrefsBtn.textContent = 'Save Team Preferences';
      }
    });
  }

  // Reset Session
  elements.resetChatBtn.addEventListener('click', () => resetSession());

  // Export Chat Transcript
  const exportChatBtn = document.getElementById('exportChatBtn');
  if (exportChatBtn) {
    exportChatBtn.addEventListener('click', async () => {
      try {
        const res = await fetch(`${API_BASE}/chat/session/${encodeURIComponent(state.sessionId)}/export?format=markdown`);
        if (!res.ok) {
          const msgs = Array.from(elements.messagesContainer.children).map(c => c.innerText.trim()).filter(Boolean);
          if (msgs.length === 0) {
            showToast('No messages in current session to export', 'info');
            return;
          }
          const mdFallback = `# Job Assistant Chat Transcript\n*Exported: ${new Date().toISOString()}*\n\n` + msgs.join('\n\n---\n\n');
          const blob = new Blob([mdFallback], { type: 'text/markdown' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `chat_transcript_${new Date().toISOString().slice(0, 10)}.md`;
          document.body.appendChild(a);
          a.click();
          a.remove();
          URL.revokeObjectURL(url);
          showToast('Chat exported as Markdown', 'success');
          return;
        }
        const text = await res.text();
        const blob = new Blob([text], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat_transcript_${new Date().toISOString().slice(0, 10)}.md`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        showToast('Chat exported as Markdown', 'success');
      } catch (err) {
        showToast(`Failed to export chat: ${err.message}`, 'error');
      }
    });
  }

  // Shortcuts Modal
  if (elements.shortcutsBtn) {
    elements.shortcutsBtn.addEventListener('click', () => openShortcutsModal());
  }
  if (elements.closeShortcutsBtn) {
    elements.closeShortcutsBtn.addEventListener('click', () => closeShortcutsModal());
  }
  if (elements.shortcutsModal) {
    elements.shortcutsModal.addEventListener('click', (e) => {
      if (e.target === elements.shortcutsModal) closeShortcutsModal();
    });
  }

  // Audit Log Modal
  const auditLogBtn = document.getElementById('auditLogBtn');
  if (auditLogBtn) {
    auditLogBtn.addEventListener('click', () => openAuditLogModal());
  }
  const closeAuditLogBtn = document.getElementById('closeAuditLogBtn');
  if (closeAuditLogBtn) {
    closeAuditLogBtn.addEventListener('click', () => closeAuditLogModal());
  }
  const auditLogModal = document.getElementById('auditLogModal');
  if (auditLogModal) {
    auditLogModal.addEventListener('click', (e) => {
      if (e.target === auditLogModal) closeAuditLogModal();
    });
  }
  const pruneAuditLogBtn = document.getElementById('pruneAuditLogBtn');
  if (pruneAuditLogBtn) {
    pruneAuditLogBtn.addEventListener('click', async () => {
      try {
        const res = await fetch(`${API_BASE}/admin/db/audit/prune`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ max_entries: 50, max_age_days: 7 })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        showToast(data.message || 'Audit log pruned', 'success');
        fetchAndRenderAuditLog();
      } catch (err) {
        showToast(`Prune failed: ${err.message}`, 'error');
      }
    });
  }

  // Global keydown shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeAllModals();
    } else if (e.key === '?' && !['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
      e.preventDefault();
      openShortcutsModal();
    } else if (e.key === '/' && !['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
      e.preventDefault();
      elements.messageInput.focus();
    }
  });

  // Global helper for quick prompts
  window.sendPrompt = (promptText) => {
    elements.messageInput.value = promptText;
    handleSendMessage();
  };

  // Global helper for 30s/60s elevator pitch generator
  window.generateElevatorPitch = async (targetRole = null) => {
    const profile = state.candidateProfile || {};
    const role = targetRole || (profile.target_roles && profile.target_roles[0]) || 'Software Engineer';
    try {
      showToast('Generating elevator pitch...', 'info');
      const res = await fetch(`${API_BASE}/chat/pitch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_title: role,
          candidate_skills: profile.skills || [],
          candidate_summary: profile.summary || ''
        })
      });
      if (res.ok) {
        const data = await res.json();
        const bulletPoints = (data.key_selling_points || []).map(p => `- ${escapeHtml(p)}`).join('\n');
        appendAssistantMessage(
          `### 🎤 Tailored Elevator Pitch: ${escapeHtml(data.job_title)}\n\n` +
          `**⏱️ 30-Second Verbal Pitch:**\n> "${escapeHtml(data.pitch_30s)}"\n\n` +
          `**⏱️ 60-Second In-Depth Pitch:**\n> "${escapeHtml(data.pitch_60s)}"\n\n` +
          `**🎯 Key Selling Points:**\n${bulletPoints}\n\n` +
          `**🗣️ Closing Call-to-Action:**\n* "${escapeHtml(data.call_to_action)}"*`
        );
        showToast('Elevator pitch generated!', 'success');
      } else {
        showToast('Failed to generate elevator pitch', 'error');
      }
    } catch (err) {
      showToast(`Error: ${err.message}`, 'error');
    }
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

  // Check for client-side /cheatsheet slash command
  if (text.toLowerCase().startsWith('/cheatsheet')) {
    const tech = text.substring(11).trim() || 'Python';
    setSending(true);
    const typingIndicatorId = showTypingIndicator();
    scrollToBottom();
    try {
      showToast(`Generating ${tech} cheatsheet...`, 'info');
      const res = await fetch(`${API_BASE}/chat/cheatsheet`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ technology: tech, seniority: 'Senior' })
      });
      removeTypingIndicator(typingIndicatorId);
      if (res.ok) {
        const data = await res.json();
        const mechanicsList = (data.core_mechanics || []).map(m => `- ${escapeHtml(m)}`).join('\n');
        const pitfallsList = (data.frequent_pitfalls || []).map(p => `- ⚠️ ${escapeHtml(p)}`).join('\n');
        const qasList = (data.high_probability_questions || []).map(q => `**Q: ${escapeHtml(q.q)}**\n> ${escapeHtml(q.a)}`).join('\n\n');
        const tipsList = (data.architecture_tips || []).map(t => `- 💡 ${escapeHtml(t)}`).join('\n');
        appendAssistantMessage(
          `### ⚡ Technical Interview Cheat-Sheet: ${escapeHtml(data.technology)} (${escapeHtml(data.seniority)})\n\n` +
          `#### ⚙️ Core Mechanics\n${mechanicsList}\n\n` +
          `#### 🚫 High-Risk Traps & Pitfalls\n${pitfallsList}\n\n` +
          `#### 🎯 Target Interview Q&As\n${qasList}\n\n` +
          `#### 🏗️ Architecture Tips\n${tipsList}`
        );
        showToast('Cheatsheet generated!', 'success');
        setSending(false);
        return;
      }
    } catch (e) {
      removeTypingIndicator(typingIndicatorId);
    }
  }

  // Check for client-side /project slash command
  if (text.toLowerCase().startsWith('/project')) {
    const proj = text.substring(8).trim() || 'Distributed Backend System';
    setSending(true);
    const typingIndicatorId = showTypingIndicator();
    scrollToBottom();
    try {
      showToast(`Generating talking points for ${proj}...`, 'info');
      const res = await fetch(`${API_BASE}/chat/project-talking-points`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_name: proj,
          tech_stack: state.candidateProfile?.skills || ['Python', 'FastAPI', 'Distributed Systems'],
          target_role: state.candidateProfile?.target_roles?.[0] || 'Senior Backend Engineer'
        })
      });
      removeTypingIndicator(typingIndicatorId);
      if (res.ok) {
        const data = await res.json();
        const archList = (data.architectural_highlights || []).map(h => `- ${escapeHtml(h)}`).join('\n');
        const tradeoffsList = (data.technical_tradeoffs || []).map(t => `- ⚖️ ${escapeHtml(t)}`).join('\n');
        const metricsList = (data.scale_and_performance_metrics || []).map(m => `- 📊 ${escapeHtml(m)}`).join('\n');
        appendAssistantMessage(
          `### 🛠️ Portfolio Deep-Dive: ${escapeHtml(data.project_name)} (${escapeHtml(data.target_role)})\n\n` +
          `#### 🗣️ Suggested Opening Pitch\n> "${escapeHtml(data.suggested_opening_pitch)}"\n\n` +
          `#### 🏗️ Key Architectural Decisions\n${archList}\n\n` +
          `#### ⚖️ Engineering Tradeoffs & Nuance\n${tradeoffsList}\n\n` +
          `#### 📈 Scale & Performance Highlights\n${metricsList}`
        );
        showToast('Project talking points generated!', 'success');
        setSending(false);
        return;
      }
    } catch (e) {
      removeTypingIndicator(typingIndicatorId);
    }
  }

  // Check for client-side /sysdesign slash command
  if (text.toLowerCase().startsWith('/sysdesign')) {
    const topic = text.substring(10).trim() || 'Distributed Rate Limiter';
    setSending(true);
    const typingIndicatorId = showTypingIndicator();
    scrollToBottom();
    try {
      showToast(`Generating system design scenario for ${topic}...`, 'info');
      const res = await fetch(`${API_BASE}/chat/system-design`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: topic,
          scale_target: '100M Daily Active Users',
          seniority: 'Senior'
        })
      });
      removeTypingIndicator(typingIndicatorId);
      if (res.ok) {
        const data = await res.json();
        const funcList = (data.functional_requirements || []).map(f => `- ${escapeHtml(f)}`).join('\n');
        const nonFuncList = (data.non_functional_requirements || []).map(n => `- ${escapeHtml(n)}`).join('\n');
        const sizingList = (data.back_of_envelope_calculations || []).map(b => `- 📐 ${escapeHtml(b)}`).join('\n');
        const compList = (data.key_architectural_components || []).map(c => `- 🧩 ${escapeHtml(c)}`).join('\n');
        const deepList = (data.deep_dive_topics || []).map(d => `- 🔍 ${escapeHtml(d)}`).join('\n');
        appendAssistantMessage(
          `### 📐 System Design Mock: ${escapeHtml(data.topic)} (${escapeHtml(data.seniority)})\n\n` +
          `#### 📋 Functional Requirements\n${funcList}\n\n` +
          `#### ⚡ Non-Functional Requirements\n${nonFuncList}\n\n` +
          `#### 🧮 Back-of-the-Envelope Estimation\n${sizingList}\n\n` +
          `#### 🏛️ Core Architectural Components\n${compList}\n\n` +
          `#### 🔬 Interviewer Deep-Dive Topics\n${deepList}`
        );
        showToast('System design scenario ready!', 'success');
        setSending(false);
        return;
      }
    } catch (e) {
      removeTypingIndicator(typingIndicatorId);
    }
  }

  // Check for client-side /resign slash command
  if (text.toLowerCase().startsWith('/resign')) {
    const company = text.substring(7).trim() || 'Current Company';
    setSending(true);
    const typingIndicatorId = showTypingIndicator();
    scrollToBottom();
    try {
      showToast(`Generating resignation notice for ${company}...`, 'info');
      const res = await fetch(`${API_BASE}/chat/resignation-letter`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          company_name: company,
          manager_name: 'Manager',
          last_working_day: 'two weeks from today',
          tenure_years: 2.0,
          tone: 'Professional and Warm',
          handover_focus: 'System documentation, project runbooks, and mentorship'
        })
      });
      removeTypingIndicator(typingIndicatorId);
      if (res.ok) {
        const data = await res.json();
        const checkList = (data.transition_checklist || []).map(c => `- ✅ ${escapeHtml(c)}`).join('\n');
        const tipList = (data.etiquette_tips || []).map(t => `- 💡 ${escapeHtml(t)}`).join('\n');
        appendAssistantMessage(
          `### 📝 Formal Resignation Notice Draft: ${escapeHtml(data.company_name)}\n\n` +
          `**Subject**: \`${escapeHtml(data.subject_line)}\`\n\n` +
          `\`\`\`text\n${escapeHtml(data.letter_body)}\n\`\`\`\n\n` +
          `#### 📋 Offboarding Handover Checklist\n${checkList}\n\n` +
          `#### 🤝 Departure Etiquette Guidelines\n${tipList}`
        );
        showToast('Resignation letter ready!', 'success');
        setSending(false);
        return;
      }
    } catch (e) {
      removeTypingIndicator(typingIndicatorId);
    }
  }

  // Check for client-side /recruiterfaq slash command
  if (text.toLowerCase().startsWith('/recruiterfaq')) {
    const company = text.substring(13).trim() || 'Tech Company';
    setSending(true);
    const typingIndicatorId = showTypingIndicator();
    scrollToBottom();
    try {
      showToast(`Generating recruiter phone screen guide for ${company}...`, 'info');
      const res = await fetch(`${API_BASE}/chat/recruiter-faq`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          role_title: 'Software Engineer',
          target_company: company,
          candidate_experience_years: 4
        })
      });
      removeTypingIndicator(typingIndicatorId);
      if (res.ok) {
        const data = await res.json();
        const faqList = (data.core_faq_questions || []).map(q => 
          `**Q: ${escapeHtml(q.question)}**\n*Evaluates*: ${escapeHtml(q.what_recruiter_evaluates)}\n*Suggested Answer*: ${escapeHtml(q.suggested_talking_points)}`
        ).join('\n\n');
        const askList = (data.questions_to_ask_recruiter || []).map(a => `- ❓ ${escapeHtml(a)}`).join('\n');
        appendAssistantMessage(
          `### 📞 Recruiter Phone Screen Prep: ${escapeHtml(data.target_company)} (${escapeHtml(data.role_title)})\n\n` +
          `#### 🎯 Core Recruiter Questions & Response Strategies\n\n${faqList}\n\n` +
          `#### 💬 Reverse Questions to Ask Recruiter\n${askList}\n\n` +
          `#### 💰 Compensation Strategy\n${escapeHtml(data.salary_discussion_strategy)}`
        );
        showToast('Recruiter screen guide ready!', 'success');
        setSending(false);
        return;
      }
    } catch (e) {
      removeTypingIndicator(typingIndicatorId);
    }
  }

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
function getJobFreshnessBadge(job) {
  const badge = job.freshness_badge || job.freshness || 'New';
  if (badge === 'New') {
    return '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">NEW</span>';
  } else if (badge === 'Recent') {
    return '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono font-semibold bg-blue-500/15 text-blue-400 border border-blue-500/30">RECENT</span>';
  }
  return '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono font-semibold bg-neutral-800 text-neutral-400 border border-neutral-700">30d+</span>';
}

function getInterviewDifficultyPill(job) {
  const title = (job.title || '').toLowerCase();
  const exp = (job.experience || '').toLowerCase();
  let tier = 'Standard';
  let colorClass = 'bg-blue-500/10 text-blue-400 border-blue-500/20';

  if (title.includes('staff') || title.includes('principal') || title.includes('director')) {
    tier = 'Rigorous';
    colorClass = 'bg-rose-500/10 text-rose-400 border-rose-500/20';
  } else if (title.includes('lead') || title.includes('senior') || title.includes('sr.') || exp.includes('senior')) {
    tier = 'Challenging';
    colorClass = 'bg-amber-500/10 text-amber-400 border-amber-500/20';
  } else if (title.includes('junior') || title.includes('intern') || title.includes('entry') || exp.includes('entry')) {
    tier = 'Accessible';
    colorClass = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
  }
  return `<span class="px-1.5 py-0.5 rounded text-[9px] font-mono border ${colorClass}" title="Estimated Interview Loop Difficulty">${tier}</span>`;
}

function getSalaryFitBadge(job) {
  const sMin = job.salary_min;
  const sMax = job.salary_max;
  if (!sMin && !sMax) return '';
  const currency = job.salary_currency || '$';
  const formatted = sMax && sMin && sMax !== sMin
    ? `${currency}${(sMin/1000).toFixed(0)}k - ${(sMax/1000).toFixed(0)}k`
    : `${currency}${((sMax || sMin)/1000).toFixed(0)}k`;
  return `<span class="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-950/50 text-emerald-300 border border-emerald-800/60 font-medium" title="Target Compensation">${formatted}</span>`;
}

function getCompanyCultureBadge(job) {
  const loc = (job.location || '').toLowerCase();
  const isRemote = loc.includes('remote') || loc.includes('anywhere') || loc.includes('distributed');
  if (isRemote) {
    return `<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-teal-500/15 text-teal-300 border border-teal-500/30" title="High autonomy remote-first culture">🌐 Remote-First</span>`;
  }
  return `<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-neutral-800 text-neutral-400 border border-neutral-700" title="Hybrid / structured office presence">🏢 Hybrid / Office</span>`;
}

function getTimezoneCompatibilityPill(job) {
  const loc = (job.location || '').toLowerCase();
  if (loc.includes('remote') || loc.includes('anywhere') || loc.includes('distributed')) {
    if (loc.includes('us') || loc.includes('americas') || loc.includes('sf') || loc.includes('nyc') || loc.includes('new york')) {
      return '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-sky-500/15 text-sky-300 border border-sky-500/30" title="Core overlap in US Timezones (EST/PST)">🕒 US TZ</span>';
    } else if (loc.includes('eu') || loc.includes('europe') || loc.includes('uk') || loc.includes('london') || loc.includes('berlin')) {
      return '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-indigo-500/15 text-indigo-300 border border-indigo-500/30" title="Core overlap in European Timezones (CET/GMT)">🕒 EU TZ</span>';
    } else if (loc.includes('apac') || loc.includes('asia') || loc.includes('india') || loc.includes('tokyo') || loc.includes('singapore')) {
      return '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-violet-500/15 text-violet-300 border border-violet-500/30" title="Core overlap in APAC Timezones (IST/JST/SGT)">🕒 APAC TZ</span>';
    }
    return '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-emerald-500/15 text-emerald-300 border border-emerald-500/30" title="Worldwide Async Remote Collaboration">🌐 Global TZ</span>';
  }
  return '';
}

function getTechModernizationBadge(job) {
  const skills = (job.skills || []).map(s => s.toLowerCase());
  const cuttingEdge = ['rust', 'go', 'fastapi', 'next.js', 'kubernetes', 'k8s', 'pytorch', 'graphql', 'tailwind'];
  const dated = ['jquery', 'svn', 'php 5', 'soap', 'jsp', 'coldfusion', 'perl'];
  const hasCuttingEdge = skills.some(s => cuttingEdge.includes(s));
  const hasDated = skills.some(s => dated.includes(s));
  if (hasCuttingEdge) {
    return '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-emerald-500/15 text-emerald-300 border border-emerald-500/30" title="Cutting-edge modern engineering stack">🚀 Modern Stack</span>';
  } else if (hasDated) {
    return '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-amber-500/15 text-amber-300 border border-amber-500/30" title="Contains legacy or dated technologies">📦 Legacy Stack</span>';
  }
  return '';
}

function getCompanyStageBadge(job) {
  const comp = (job.company || '').toLowerCase();
  const desc = (job.description || '').toLowerCase();
  if (comp.includes('startup') || comp.includes('labs') || desc.includes('seed') || desc.includes('series a') || desc.includes('founding')) {
    return '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-purple-500/15 text-purple-300 border border-purple-500/30" title="Early Stage Startup">🌱 Startup</span>';
  } else if (comp.includes('google') || comp.includes('microsoft') || comp.includes('amazon') || comp.includes('apple') || comp.includes('meta') || desc.includes('enterprise') || desc.includes('fortune')) {
    return '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-blue-500/15 text-blue-300 border border-blue-500/30" title="Enterprise Scale Organization">🏛️ Enterprise</span>';
  }
  return '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-teal-500/15 text-teal-300 border border-teal-500/30" title="Growth / Scaling Stage">📈 Growth</span>';
}

function getJobClarityBadge(job) {
  const desc = (job.description || '').toLowerCase();
  const buzzwords = ['wear multiple hats', 'rockstar', 'ninja', 'fast-paced environment', 'other duties as assigned', 'work hard play hard'];
  const hasBuzz = buzzwords.some(b => desc.includes(b));
  if (hasBuzz) {
    return '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-amber-500/10 text-amber-400 border border-amber-500/20" title="Contains ambiguous job description buzzwords">⚠️ Ambiguous JD</span>';
  } else if (desc.length > 200 && (desc.includes('responsibilities') || desc.includes('requirements'))) {
    return '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" title="Well-structured requirements with high clarity">✨ Clear JD</span>';
  }
  return '';
}

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
      <div class="flex items-center justify-between mb-2">
        <div class="text-[10px] font-mono uppercase tracking-wider text-neutral-400 font-semibold">Assistant</div>
        <button
          class="copy-msg-btn px-2 py-0.5 rounded text-[10px] font-mono text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800 transition-colors flex items-center gap-1"
          title="Copy message to clipboard"
        >
          <svg class="w-3 h-3 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/>
          </svg>
          <span class="btn-text">Copy</span>
        </button>
      </div>
      <div class="prose-chat">${parsedHtml}</div>
      <div class="actions-slot mt-3 space-y-3"></div>
    </div>
  `;

  const copyBtn = msgEl.querySelector('.copy-msg-btn');
  if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(markdownContent || '');
        const btnText = copyBtn.querySelector('.btn-text');
        if (btnText) btnText.textContent = 'Copied!';
        setTimeout(() => {
          if (btnText) btnText.textContent = 'Copy';
        }, 2000);
      } catch (err) {
        showToast('Failed to copy message', 'error');
      }
    });
  }

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
            ${getJobFreshnessBadge(job)}
            ${getInterviewDifficultyPill(job)}
            ${getCompanyCultureBadge(job)}
            ${getTimezoneCompatibilityPill(job)}
            ${getTechModernizationBadge(job)}
            ${getCompanyStageBadge(job)}
            ${getJobClarityBadge(job)}
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
        ${getSalaryFitBadge(job)}
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
            onclick="requestRoadmapForJob('${escapeHtml(job.job_id)}', '${escapeHtml(job.title)}')"
            title="Generate custom learning roadmap for missing skills"
            class="px-2.5 py-1.5 rounded text-xs font-medium text-neutral-300 hover:text-white bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 transition-colors"
          >
            Roadmap
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
    let jobs = data.jobs || [];

    if (state.minSalary > 0) {
      jobs = jobs.filter(j => (j.salary_max || j.salary_min || 0) >= state.minSalary);
    }

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
            ${getJobFreshnessBadge(job)}
            ${getInterviewDifficultyPill(job)}
            ${getCompanyCultureBadge(job)}
            ${getTimezoneCompatibilityPill(job)}
            ${getTechModernizationBadge(job)}
            ${getCompanyStageBadge(job)}
            ${getJobClarityBadge(job)}
            <span class="text-[10px] font-mono px-1.5 py-0.2 rounded bg-neutral-800 text-neutral-400 border border-neutral-700">${escapeHtml(job.experience || 'All Levels')}</span>
          </div>
          <p class="text-xs text-neutral-400 mt-0.5">${escapeHtml(job.company)} • <span class="text-neutral-300">${escapeHtml(job.location)}</span></p>
          <div class="flex flex-wrap items-center gap-1.5 mt-2">${getSalaryFitBadge(job)}${skillPills}</div>
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

window.requestRoadmapForJob = (jobId, title) => {
  elements.messageInput.value = `/roadmap ${jobId}`;
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
    let apps = await res.json();

    const totalCount = apps.length;
    if (state.appStatusFilter && state.appStatusFilter !== 'ALL') {
      apps = apps.filter(a => (a.status || 'Applied').toLowerCase() === state.appStatusFilter.toLowerCase());
    }

    elements.appsModalSubtitle.innerText = `${totalCount} total applications submitted (${apps.length} showing)`;
    updateApplicationsBadge(totalCount);

    // Fetch priority queue telemetry
    const priorityBanner = document.getElementById('appsPriorityBanner');
    const priorityBannerText = document.getElementById('appsPriorityBannerText');
    const priorityMap = new Map();
    try {
      const qRes = await fetch(`${API_BASE}/applications/queue/prioritized`);
      if (qRes.ok) {
        const qData = await qRes.json();
        (qData.queue || []).forEach(item => priorityMap.set(item.job_id, item));
        if (priorityBanner && priorityBannerText) {
          if (qData.critical_actions_count > 0) {
            priorityBannerText.innerText = `${qData.critical_actions_count} urgent ${qData.critical_actions_count === 1 ? 'application requires' : 'applications require'} immediate attention`;
            priorityBanner.classList.remove('hidden');
          } else {
            priorityBanner.classList.add('hidden');
          }
        }
      }
    } catch (e) {}

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
      const questionCount = Array.isArray(app.interview_questions) ? app.interview_questions.length : 0;
      const questionBadge = questionCount > 0
        ? `<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-purple-500/15 text-purple-300 border border-purple-500/30">🎯 ${questionCount} Questions</span>`
        : '';
      const difficultyBadge = app.interview_difficulty_tier
        ? `<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-amber-500/15 text-amber-300 border border-amber-500/30">${escapeHtml(app.interview_difficulty_tier)} (${app.interview_difficulty_score || 3}/5)</span>`
        : '';
      const thankYouBadge = app.thank_you_sent
        ? `<button onclick="toggleThankYouStatus('${escapeHtml(app.job_id)}', false)" title="Thank-you note recorded. Click to mark pending." class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/25 transition-colors">✉️ Thank-You Sent</button>`
        : (['Interview', 'Offer', 'Under Review'].includes(currentStatus)
          ? `<button onclick="toggleThankYouStatus('${escapeHtml(app.job_id)}', true)" title="Click to record thank-you note sent" class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-neutral-900 text-neutral-400 hover:text-neutral-200 border border-neutral-800 hover:border-neutral-700 transition-colors">✉️ Send Thank-You</button>`
          : '');
      const priorityInfo = priorityMap.get(app.job_id);
      const urgencyBadge = priorityInfo && priorityInfo.urgency_tier === 'Critical'
        ? `<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-rose-500/15 text-rose-300 border border-rose-500/30" title="${escapeHtml(priorityInfo.recommended_action)}">⚡ Urgent Action</span>`
        : '';
      const todos = Array.isArray(app.todo_checklist) ? app.todo_checklist : [];
      const completedTodos = todos.filter(t => t.completed).length;
      const todoBadge = todos.length > 0
        ? `<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-neutral-800 text-neutral-300 border border-neutral-700" title="${completedTodos} of ${todos.length} tasks completed">✓ ${completedTodos}/${todos.length} Tasks</span>`
        : '';
      const addTodoBtn = `<button onclick="promptAddApplicationTodo('${escapeHtml(app.job_id)}')" title="Add checklist task" class="px-1.5 py-0.5 rounded text-[9px] font-mono text-neutral-400 hover:text-neutral-200 bg-neutral-900 border border-neutral-800 hover:border-neutral-700 transition-colors">+ Task</button>`;
      
      let todosHtml = '';
      if (todos.length > 0) {
        const todoItems = todos.slice(-3).map(t =>
          `<div class="flex items-center gap-1.5 text-[11px] text-neutral-300">
            <input type="checkbox" ${t.completed ? 'checked' : ''} onchange="toggleApplicationTodoItem('${escapeHtml(app.job_id)}', '${escapeHtml(t.id)}', this.checked)" class="w-3 h-3 rounded bg-neutral-800 border-neutral-700 text-emerald-500 focus:ring-0 cursor-pointer">
            <span class="${t.completed ? 'line-through text-neutral-500' : ''}">${escapeHtml(t.task)}</span>
          </div>`
        ).join('');
        todosHtml = `<div class="mt-2 pt-2 border-t border-neutral-900 space-y-1">${todoItems}</div>`;
      }

      const debriefInfo = app.debrief_reflection;
      const debriefBadge = currentStatus === 'Rejected'
        ? (debriefInfo
          ? `<button onclick="promptLogApplicationDebrief('${escapeHtml(app.job_id)}')" title="Debrief logged for ${escapeHtml(debriefInfo.stage_reached || 'stage')}. Click to update." class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-purple-500/15 text-purple-300 border border-purple-500/30 hover:bg-purple-500/25 transition-colors">📝 Debrief Logged</button>`
          : `<button onclick="promptLogApplicationDebrief('${escapeHtml(app.job_id)}')" title="Log lessons learned and reflection notes" class="px-1.5 py-0.5 rounded text-[9px] font-mono text-neutral-400 hover:text-purple-300 bg-neutral-900 border border-neutral-800 hover:border-neutral-700 transition-colors">📝 Add Debrief</button>`)
        : '';
      const refInfo = app.referral;
      const referralBadge = refInfo
        ? `<button onclick="promptLogReferral('${escapeHtml(app.job_id)}')" title="Referred by ${escapeHtml(refInfo.referrer_name)} (${escapeHtml(refInfo.referral_status || 'Submitted')}). Click to edit." class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-500/25 transition-colors">🤝 Referred: ${escapeHtml(refInfo.referrer_name)}</button>`
        : `<button onclick="promptLogReferral('${escapeHtml(app.job_id)}')" title="Track employee referral contact" class="px-1.5 py-0.5 rounded text-[9px] font-mono text-neutral-400 hover:text-indigo-300 bg-neutral-900 border border-neutral-800 hover:border-neutral-700 transition-colors">🤝 + Referral</button>`;
      const offerBadge = currentStatus === 'Offer'
        ? `<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">🎉 Offer Received</span>`
        : '';
      const appTags = Array.isArray(app.tags) ? app.tags : [];
      const tagChips = appTags.map(t =>
        `<span class="inline-flex items-center gap-1 px-1.5 py-0.2 rounded text-[9px] font-mono bg-neutral-900 text-neutral-300 border border-neutral-800">
          <span>🏷️ ${escapeHtml(t)}</span>
          <button onclick="removeAppCustomTag('${escapeHtml(app.job_id)}', '${escapeHtml(t)}')" class="hover:text-rose-400 font-bold ml-0.5" title="Remove tag">×</button>
        </span>`
      ).join(' ');
      const addTagBtn = `<button onclick="promptAddAppCustomTag('${escapeHtml(app.job_id)}')" class="px-1.5 py-0.2 rounded text-[9px] font-mono text-neutral-400 hover:text-white bg-neutral-900 border border-neutral-800 hover:border-neutral-700 transition-colors">+ Tag</button>`;

      const statusOptions = statuses.map(s =>
        `<option value="${s}" ${s === currentStatus ? 'selected' : ''}>${s}</option>`
      ).join('');
      card.innerHTML = `
        <div class="flex-1">
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-xs font-mono text-neutral-400">#${idx + 1}</span>
            <h4 class="text-xs sm:text-sm font-semibold text-white">${escapeHtml(app.job_title || 'Software Role')}</h4>
            ${urgencyBadge}
            ${questionBadge}
            ${difficultyBadge}
            ${referralBadge}
            ${thankYouBadge}
            ${offerBadge}
            ${debriefBadge}
            ${todoBadge}
            ${addTodoBtn}
          </div>
          <p class="text-xs text-neutral-400 mt-0.5">${escapeHtml(app.company || 'Tech Company')} • <span class="text-neutral-300">${escapeHtml(app.location || 'Remote')}</span></p>
          <div class="flex flex-wrap items-center gap-1 mt-1.5">${tagChips} ${addTagBtn}</div>
          <p class="text-[11px] font-mono text-neutral-500 mt-1">Submitted on ${escapeHtml(app.applied_date || 'Recent')}</p>
          ${todosHtml}
        </div>
        <div class="text-right flex-shrink-0 flex flex-col items-end gap-2">
          <select
            onchange="updateApplicationStatusFromModal('${escapeHtml(app.job_id)}', this.value)"
            aria-label="Update application status"
            class="px-2 py-0.5 rounded text-[11px] font-mono font-medium bg-neutral-800 text-neutral-200 border border-neutral-700 focus:outline-none focus:border-neutral-500 cursor-pointer"
          >
            ${statusOptions}
          </select>
          <div class="flex items-center gap-1.5">
            ${currentStatus === 'Offer' ? `
              <button
                onclick="openCounterOfferCalculator('${escapeHtml(app.job_id)}', '${escapeHtml(app.company || 'Company')}', '${escapeHtml(app.job_title || 'Role')}')"
                title="Calculate data-backed counter-offer package"
                class="px-2 py-0.5 rounded text-[10px] text-emerald-300 hover:text-white bg-emerald-950/60 hover:bg-emerald-900 border border-emerald-800/80 transition-colors"
              >
                💰 Counter-Offer
              </button>
            ` : ''}
            <button
              onclick="rollbackApplicationFromModal('${escapeHtml(app.job_id)}')"
              title="Revert to previous status snapshot"
              class="px-2 py-0.5 rounded text-[10px] text-neutral-400 hover:text-amber-300 hover:bg-amber-950/40 border border-neutral-800 hover:border-amber-700/50 transition-colors"
            >
              ⤺ Undo
            </button>
            <button
              onclick="withdrawApplicationFromModal('${escapeHtml(app.job_id)}')"
              title="Withdraw this application"
              class="px-2 py-0.5 rounded text-[10px] text-neutral-400 hover:text-rose-400 hover:bg-neutral-900 border border-neutral-800 hover:border-neutral-700 transition-colors"
            >
              Withdraw
            </button>
          </div>
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

async function fetchAndUpdateStreakMetrics() {
  try {
    const res = await fetch(`${API_BASE}/applications/metrics/streak`);
    if (!res.ok) return;
    const data = await res.json();
    if (elements.navStreakCountText) {
      elements.navStreakCountText.textContent = `${data.current_streak_days}d streak`;
    }
    if (elements.navDailyGoalText) {
      elements.navDailyGoalText.textContent = `${data.submitted_today}/${data.daily_goal} today`;
      if (data.goal_met_today) {
        elements.navDailyGoalText.className = 'text-emerald-400 font-semibold';
      } else {
        elements.navDailyGoalText.className = 'text-neutral-400';
      }
    }
  } catch (e) {}
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
    fetchAndUpdateStreakMetrics();
  } catch (e) {
    // Silently ignore if offline
  }
}

/**
 * Diagnostics Modal
 */
async function openDiagnosticsModal() {
  elements.diagSessionId.innerText = state.sessionId;
  elements.diagAppsCount.innerText = `${state.appliedJobIds.size} tracked in session`;
  elements.diagnosticsModal.classList.remove('hidden');
  fetchSystemHealth();

  const diskSpeedEl = document.getElementById('diagDiskReadSpeed');
  const integrityEl = document.getElementById('diagDbIntegrity');
  try {
    const res = await fetch(`${API_BASE}/db/diagnostics`);
    if (res.ok) {
      const data = await res.json();
      if (diskSpeedEl && data.storage_read_benchmark) {
        diskSpeedEl.innerText = `${data.storage_read_benchmark.read_megabytes_per_sec.toFixed(1)} MB/s (${data.storage_read_benchmark.benchmark_duration_ms.toFixed(1)}ms)`;
      }
      if (integrityEl && data.collection_checksums) {
        integrityEl.innerText = `OK (${data.collection_checksums.length} collections)`;
      }
    }
  } catch (err) {
    if (diskSpeedEl) diskSpeedEl.innerText = 'Offline';
  }
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

  // Render language tags
  renderDrawerLanguages(p.languages);

  // Render roles
  elements.drawerRoles.innerHTML = (p.preferred_roles || ['Software Engineer']).map(r =>
    `<span class="px-2 py-0.5 rounded bg-neutral-800 text-xs text-neutral-300 border border-neutral-700">${escapeHtml(r)}</span>`
  ).join('');

  // Render portfolio links
  renderDrawerLinks(p.portfolio_links);

  // Update Profile Completeness
  updateProfileCompletenessDrawer(p);

  // Fetch Team and Stage Preferences
  fetch(`${API_BASE}/profile/team-preferences`)
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (data && elements.drawerTeamSizeSelect) {
        elements.drawerTeamSizeSelect.value = data.preferred_team_size || '10-50';
      }
      if (data && elements.drawerCompanyStageSelect) {
        elements.drawerCompanyStageSelect.value = data.preferred_company_stage || 'Growth';
      }
    })
    .catch(() => {});

  // Fetch Availability Preferences
  fetch(`${API_BASE}/profile/availability`)
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (data && elements.drawerNoticeBadge) {
        elements.drawerNoticeBadge.textContent = `${data.notice_period_days} Days Notice`;
      }
      if (data && elements.drawerStartDate) {
        elements.drawerStartDate.textContent = data.earliest_start_date ? `Earliest: ${data.earliest_start_date}` : (data.is_immediately_available ? 'Available Immediately' : 'Flexible Start');
      }
    })
    .catch(() => {});

  elements.profileDrawer.classList.remove('hidden');
}

function renderDrawerLanguages(languages) {
  if (!elements.drawerLanguagesTags) return;
  const langs = (languages && languages.length > 0) 
    ? languages 
    : [{ language: 'English', proficiency: 'Professional Working' }];
  elements.drawerLanguagesTags.innerHTML = langs.map(l =>
    `<span class="px-2 py-0.5 rounded bg-neutral-800 text-xs text-neutral-200 border border-neutral-700 flex items-center gap-1.5">
      <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
      ${escapeHtml(l.language)} <span class="text-[10px] text-neutral-400 font-mono">(${escapeHtml(l.proficiency)})</span>
    </span>`
  ).join('');
}

function renderDrawerLinks(links) {
  if (!elements.drawerLinksList) return;
  if (!links || links.length === 0) {
    elements.drawerLinksList.innerHTML = '<p class="text-xs text-neutral-400 italic">No external portfolio links saved yet.</p>';
    return;
  }
  elements.drawerLinksList.innerHTML = links.map(l =>
    `<a href="${escapeHtml(l.url)}" target="_blank" rel="noopener noreferrer" 
        class="flex items-center justify-between p-2 rounded bg-neutral-950 hover:bg-neutral-800 border border-neutral-800 text-xs text-neutral-200 transition-colors group">
      <span class="flex items-center gap-2 font-medium">
        <svg class="w-3.5 h-3.5 text-neutral-400 group-hover:text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
        </svg>
        ${escapeHtml(l.label)}
      </span>
      <span class="text-[10px] text-neutral-400 truncate max-w-[150px] font-mono">${escapeHtml(l.url.replace(/^https?:\/\//, ''))}</span>
    </a>`
  ).join('');
}



async function updateProfileCompletenessDrawer(profile) {
  try {
    const res = await fetch('/api/profile/completeness');
    if (res.ok) {
      const data = await res.json();
      const score = Math.round(data.completeness_score || 0);
      renderCompleteness(score, data.strength_rating || 'Incomplete');
      return;
    }
  } catch (err) {
    console.debug('Failed to fetch completeness endpoint, falling back to local calculation');
  }

  // Local fallback calculation
  let score = 20; // Base score for having an uploaded document
  if (profile.name && profile.name !== 'Candidate Profile') score += 20;
  if (profile.email && profile.email.includes('@')) score += 15;
  if (profile.skills && profile.skills.length >= 3) score += 25;
  if (profile.summary && profile.summary.length > 20) score += 20;
  score = Math.min(score, 100);

  const rating = score >= 85 ? 'Complete' : score >= 65 ? 'Strong' : score >= 40 ? 'Moderate' : 'Incomplete';
  renderCompleteness(score, rating);
}

function renderCompleteness(score, rating) {
  if (!elements.drawerCompletenessBar || !elements.drawerCompletenessBadge) return;
  elements.drawerCompletenessBar.style.width = `${score}%`;
  elements.drawerCompletenessBadge.innerText = `${score}% (${rating})`;

  if (score >= 80) {
    elements.drawerCompletenessBar.className = 'bg-emerald-500 h-2 rounded-full transition-all duration-500';
    elements.drawerCompletenessBadge.className = 'font-semibold text-emerald-400 font-mono';
    elements.drawerCompletenessHint.innerText = 'Excellent! Your profile has high ATS readiness and strength.';
  } else if (score >= 50) {
    elements.drawerCompletenessBar.className = 'bg-amber-500 h-2 rounded-full transition-all duration-500';
    elements.drawerCompletenessBadge.className = 'font-semibold text-amber-400 font-mono';
    elements.drawerCompletenessHint.innerText = 'Good baseline. Add more skills and details to boost your score.';
  } else {
    elements.drawerCompletenessBar.className = 'bg-rose-500 h-2 rounded-full transition-all duration-500';
    elements.drawerCompletenessBadge.className = 'font-semibold text-rose-400 font-mono';
    elements.drawerCompletenessHint.innerText = 'Upload a detailed resume to unlock personalized matching.';
  }
}


function closeProfileDrawer() {
  elements.profileDrawer.classList.add('hidden');
}

function openShortcutsModal() {
  if (elements.shortcutsModal) elements.shortcutsModal.classList.remove('hidden');
}

function closeShortcutsModal() {
  if (elements.shortcutsModal) elements.shortcutsModal.classList.add('hidden');
}

function closeAllModals() {
  closeJobsModal();
  closeApplicationsModal();
  closeDiagnosticsModal();
  closeProfileDrawer();
  closeShortcutsModal();
  closeAuditLogModal();
}

function openAuditLogModal() {
  const modal = document.getElementById('auditLogModal');
  if (modal) {
    modal.classList.remove('hidden');
    fetchAndRenderAuditLog();
  }
}

function closeAuditLogModal() {
  const modal = document.getElementById('auditLogModal');
  if (modal) modal.classList.add('hidden');
}

async function fetchAndRenderAuditLog() {
  const container = document.getElementById('auditListContainer');
  const subtitle = document.getElementById('auditModalSubtitle');
  if (!container) return;

  container.innerHTML = '<div class="p-8 text-center text-xs text-neutral-500 font-mono">Loading audit events...</div>';
  try {
    const res = await fetch(`${API_BASE}/admin/db/audit-log?limit=50`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const entries = data.entries || [];
    if (subtitle) subtitle.innerText = `${data.total_entries || 0} recorded transaction events`;

    if (entries.length === 0) {
      container.innerHTML = '<div class="p-8 text-center text-xs text-neutral-500">No audit entries recorded yet.</div>';
      return;
    }

    container.innerHTML = '';
    entries.forEach(entry => {
      const card = document.createElement('div');
      card.className = 'p-2.5 rounded bg-neutral-950 border border-neutral-800 text-xs flex items-center justify-between gap-3';
      const actionBadge = `<span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold bg-neutral-800 text-neutral-300 border border-neutral-700">${escapeHtml(entry.action || 'TRANSACTION')}</span>`;
      card.innerHTML = `
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2">
            ${actionBadge}
            <span class="text-neutral-300 truncate">${escapeHtml(entry.details || '')}</span>
          </div>
          <div class="text-[10px] text-neutral-500 font-mono mt-1">Entity: ${escapeHtml(entry.entity || 'db')} • ID: ${escapeHtml(entry.entity_id || '')}</div>
        </div>
        <span class="text-[10px] font-mono text-neutral-500 flex-shrink-0">${escapeHtml((entry.timestamp || '').slice(0, 19).replace('T', ' '))}</span>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    container.innerHTML = `<div class="p-6 text-center text-xs text-rose-400">Failed to load audit log: ${escapeHtml(err.message)}</div>`;
  }
}

window.rollbackApplicationFromModal = async (jobId) => {
  try {
    const res = await fetch(`${API_BASE}/applications/${encodeURIComponent(jobId)}/rollback`, {
      method: 'POST'
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Rollback failed (${res.status})`);
    }
    const data = await res.json();
    showToast(`Reverted ${jobId} to ${data.restored_status || 'previous status'}`, 'success');
    fetchAndRenderApplicationsList();
  } catch (err) {
    showToast(`Rollback unavailable: ${err.message}`, 'error');
  }
};

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

window.toggleThankYouStatus = async (jobId, markSent) => {
  try {
    const res = await fetch(`${API_BASE}/applications/${encodeURIComponent(jobId)}/thank-you`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thank_you_sent: markSent })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to update status (${res.status})`);
    }
    showToast(markSent ? 'Marked thank-you note as sent!' : 'Thank-you note marked as pending', 'success');
    fetchAndRenderApplicationsList();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
};

window.promptAddApplicationTodo = async (jobId) => {
  const task = prompt('Enter a new checklist task for this application:');
  if (!task || !task.trim()) return;
  try {
    const res = await fetch(`${API_BASE}/applications/${encodeURIComponent(jobId)}/todos`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task: task.trim() })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to add task (${res.status})`);
    }
    showToast('Checklist task added!', 'success');
    fetchAndRenderApplicationsList();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
};

window.toggleApplicationTodoItem = async (jobId, todoId, completed) => {
  try {
    const res = await fetch(`${API_BASE}/applications/${encodeURIComponent(jobId)}/todos/${encodeURIComponent(todoId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ completed: Boolean(completed) })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to update task (${res.status})`);
    }
    showToast(completed ? 'Task completed' : 'Task pending', 'info');
    fetchAndRenderApplicationsList();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
};

window.openCounterOfferCalculator = async (jobId, company, role) => {
  const baseInput = prompt(`Enter annual base salary offered for ${role} at ${company} ($):`, "150000");
  if (!baseInput) return;
  const base = parseFloat(baseInput.replace(/[^0-9.]/g, ''));
  if (isNaN(base) || base <= 0) {
    showToast('Invalid base salary entered', 'error');
    return;
  }
  const equityInput = prompt('Enter equity value offered ($ optional, e.g. 30000):', "30000");
  const equity = equityInput ? parseFloat(equityInput.replace(/[^0-9.]/g, '')) || 0 : 0;

  try {
    showToast('Calculating counter-offer proposal...', 'info');
    const res = await fetch(`${API_BASE}/applications/${encodeURIComponent(jobId)}/counter-offer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        base_salary_offered: base,
        equity_offered: equity,
        target_strategy: 'balanced'
      })
    });
    if (res.ok) {
      const data = await res.json();
      closeApplicationsModal();
      const pointsList = (data.negotiation_talking_points || []).map(p => `- ${escapeHtml(p)}`).join('\n');
      appendAssistantMessage(
        `### 💰 Data-Backed Counter-Offer Strategy: ${escapeHtml(company)} — ${escapeHtml(role)}\n\n` +
        `| Component | Offered | Recommended Counter |\n` +
        `|---|---|---|\n` +
        `| **Base Salary** | $${data.original_base.toLocaleString()} | **$${data.recommended_counter_base.toLocaleString()}** |\n` +
        `| **Equity Target** | $${(equity || 0).toLocaleString()} | **$${data.recommended_equity.toLocaleString()}** |\n` +
        `| **Target Signing Bonus** | $0 | **$${data.recommended_sign_on.toLocaleString()}** |\n` +
        `| **Total First Year Comp** | $${(base + equity).toLocaleString()} | **$${data.total_counter_comp_value.toLocaleString()}** (+${data.percentage_increase}%) |\n\n` +
        `#### 🎯 Key Negotiation Leverage Points\n${pointsList}\n\n` +
        `#### ✉️ Suggested Response Email Script\n> "${escapeHtml(data.suggested_email_opening)}"`
      );
      showToast('Counter-offer proposal generated in chat!', 'success');
    } else {
      showToast('Failed to generate counter-proposal', 'error');
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
};

window.promptLogApplicationDebrief = async (jobId) => {
  const stage = prompt('Furthest interview stage reached (e.g. Phone Screen, Technical Screen, System Design, Onsite):', 'Technical Screen');
  if (!stage) return;
  const lessons = prompt('Key lessons learned / things to improve next time (comma separated):', 'Clarify performance constraints earlier, Practice distributed caching');
  if (!lessons) return;
  const notes = prompt('Any interviewer feedback or personal reflection notes:', '');

  try {
    const res = await fetch(`${API_BASE}/applications/${encodeURIComponent(jobId)}/debrief`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        stage_reached: stage.trim(),
        lessons_learned: lessons.split(',').map(s => s.trim()).filter(Boolean),
        reflection_notes: notes ? notes.trim() : null
      })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to log debrief (${res.status})`);
    }
    showToast('Debrief reflection recorded!', 'success');
    fetchAndRenderApplicationsList();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
};

window.promptLogReferral = async (jobId) => {
  const name = prompt('Enter referrer name (e.g. Alex Johnson):');
  if (!name || !name.trim()) return;
  const email = prompt('Enter referrer email (optional):', '');
  const status = prompt('Referral status (Requested, Submitted, Confirmed, Declined):', 'Submitted');
  const relationship = prompt('Relationship (e.g. Former Colleague, College Alumni, LinkedIn Connection):', 'Former Colleague');

  try {
    const res = await fetch(`${API_BASE}/applications/${encodeURIComponent(jobId)}/referral`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        referrer_name: name.trim(),
        referrer_email: email ? email.trim() : null,
        referral_status: status ? status.trim() : 'Submitted',
        relationship: relationship ? relationship.trim() : 'Former Colleague'
      })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to log referral (${res.status})`);
    }
    showToast('Referral details saved!', 'success');
    fetchAndRenderApplicationsList();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
};

window.promptAddAppCustomTag = async (jobId) => {
  const tag = prompt('Enter a custom tag or category label for this application (e.g. Dream Job, Referred, High Priority):');
  if (!tag || !tag.trim()) return;
  try {
    const res = await fetch(`${API_BASE}/applications/${encodeURIComponent(jobId)}/tags/custom`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tag: tag.trim() })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to add tag (${res.status})`);
    }
    showToast('Custom tag added!', 'success');
    fetchAndRenderApplicationsList();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
};

window.removeAppCustomTag = async (jobId, tag) => {
  try {
    const res = await fetch(`${API_BASE}/applications/${encodeURIComponent(jobId)}/tags/custom/${encodeURIComponent(tag)}`, {
      method: 'DELETE'
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to remove tag (${res.status})`);
    }
    showToast(`Tag '${tag}' removed`, 'info');
    fetchAndRenderApplicationsList();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
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

    try {
      const pingRes = await fetch(`${API_BASE}/system/ping`);
      if (pingRes.ok) {
        const pingData = await pingRes.json();
        if (elements.diagApiStatus) {
          elements.diagApiStatus.innerText = `Online (${pingData.uptime_human} uptime)`;
        }
      }
    } catch (e) {}
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
