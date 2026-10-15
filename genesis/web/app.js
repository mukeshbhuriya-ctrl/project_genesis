/**
 * PROJECT GENESIS — FRONTEND CONTROLLER
 */

document.addEventListener('DOMContentLoaded', () => {
  // ── State ──
  const state = {
    activeTab: 'chat',
    activeConversationId: null,
    models: [],
    selectedModel: 'qwen2.5:7b',
    searchMode: 'hybrid',
    selectedCategory: '',
  };

  // ── Tab Navigation ──
  const tabs = document.querySelectorAll('.nav-tab');
  const panes = document.querySelectorAll('.tab-content');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      state.activeTab = target;

      tabs.forEach(t => t.classList.toggle('active', t === tab));
      panes.forEach(p => p.classList.toggle('active', p.id === `pane-${target}`));

      // Refresh data on tab switch
      if (target === 'memory') loadMemories();
      if (target === 'knowledge') loadDocuments();
      if (target === 'tasks') { loadTools(); loadTasks(); }
      if (target === 'evaluation') { loadBenchmarks(); loadEvalRuns(); }
      if (target === 'training') loadTrainingData();
      if (target === 'research') loadResearchData();
      if (target === 'hardware') loadHardwareAndModels();
      if (target === 'backups') loadBackups();
    });
  });

  // ── Chat Subsystem ──
  const chatMessagesStream = document.getElementById('chat-messages-stream');
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  const btnChatSend = document.getElementById('btn-chat-send');
  const conversationsList = document.getElementById('conversations-list');
  const btnNewChat = document.getElementById('btn-new-chat');
  const chatModelSelect = document.getElementById('chat-model-select');
  const activeChatTitle = document.getElementById('active-chat-title');
  const activeModelTag = document.getElementById('active-model-tag');
  const chatTemp = document.getElementById('chat-temp');
  const chatTempVal = document.getElementById('chat-temp-val');
  const tokenMetrics = document.getElementById('token-metrics');

  chatTemp.addEventListener('input', () => {
    chatTempVal.textContent = chatTemp.value;
  });

  // Auto-resize chat textarea
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + 'px';
  });

  // Enter sends, Shift+Enter new line
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event('submit'));
    }
  });

  // Quick prompt chips
  document.querySelectorAll('.quick-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      chatInput.value = chip.dataset.prompt;
      chatInput.dispatchEvent(new Event('input'));
      chatForm.dispatchEvent(new Event('submit'));
    });
  });

  // Load conversations list
  async function loadConversations() {
    try {
      const res = await fetch('/api/conversations');
      const convs = await res.json();
      conversationsList.innerHTML = '';

      if (!convs || convs.length === 0) {
        conversationsList.innerHTML = '<div class="loading-spinner">No conversations yet</div>';
        return;
      }

      convs.forEach(c => {
        const item = document.createElement('div');
        item.className = `conversation-item ${c.id === state.activeConversationId ? 'active' : ''}`;
        item.textContent = c.title || 'Untitled Dialogue';
        item.title = c.title;
        item.addEventListener('click', () => selectConversation(c.id, c.title));
        conversationsList.appendChild(item);
      });
    } catch (e) {
      conversationsList.innerHTML = '<div class="loading-spinner">Failed to load dialogues</div>';
    }
  }

  async function selectConversation(id, title) {
    state.activeConversationId = id;
    activeChatTitle.textContent = title || 'Active Conversation';
    loadConversations();
    chatMessagesStream.innerHTML = '<div class="loading-spinner">Loading message history...</div>';

    try {
      const res = await fetch(`/api/conversations/${id}/messages`);
      const msgs = await res.json();
      chatMessagesStream.innerHTML = '';

      msgs.forEach(m => {
        appendMessageBubble(m.role, m.content, {
          model: m.model_id,
          tokens: m.token_count,
          createdAt: m.created_at,
        });
      });
      scrollChatToBottom();
    } catch (e) {
      chatMessagesStream.innerHTML = '<div class="loading-spinner">Failed to load conversation history</div>';
    }
  }

  btnNewChat.addEventListener('click', () => {
    state.activeConversationId = null;
    activeChatTitle.textContent = 'New Conversation';
    chatMessagesStream.innerHTML = `
      <div class="welcome-card">
        <div class="welcome-icon">🧬</div>
        <h3>Welcome to Project Genesis</h3>
        <p>A self-improving, local-first AI platform running directly on your dedicated hardware.</p>
        <div class="quick-prompts">
          <button class="quick-chip" data-prompt="Explain the design architecture of Project Genesis.">Design Architecture</button>
          <button class="quick-chip" data-prompt="What hardware optimizations are enabled for local inference?">Hardware Optimizations</button>
          <button class="quick-chip" data-prompt="How does semantic memory retrieval work in Phase 2?">Memory &amp; Embeddings</button>
        </div>
      </div>
    `;
    loadConversations();
  });

  function appendMessageBubble(role, content, meta = {}) {
    // Remove welcome card if present
    const welcome = chatMessagesStream.querySelector('.welcome-card');
    if (welcome) welcome.remove();

    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'U' : 'G';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = content;

    if (role === 'assistant' && (meta.tokens || meta.model)) {
      const metaEl = document.createElement('div');
      metaEl.className = 'message-meta';
      metaEl.textContent = `${meta.model || state.selectedModel} • ${meta.tokens || 0} tokens`;
      bubble.appendChild(metaEl);
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatMessagesStream.appendChild(row);
    scrollChatToBottom();
    return bubble;
  }

  function scrollChatToBottom() {
    chatMessagesStream.scrollTop = chatMessagesStream.scrollHeight;
  }

  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const content = chatInput.value.trim();
    if (!content) return;

    chatInput.value = '';
    chatInput.style.height = 'auto';
    btnChatSend.disabled = true;

    // Append User Message
    appendMessageBubble('user', content);

    // Create placeholder for assistant response
    const assistantBubble = appendMessageBubble('assistant', 'Generating response via GPU...');

    try {
      const payload = {
        messages: [{ role: 'user', content }],
        model: state.selectedModel,
        conversation_id: state.activeConversationId,
        temperature: parseFloat(chatTemp.value),
      };

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Chat error (${res.status}): ${await res.text()}`);
      }

      const data = await res.json();
      state.activeConversationId = data.conversation_id;

      // Update bubble with actual response
      assistantBubble.textContent = data.message.content;
      const metaEl = document.createElement('div');
      metaEl.className = 'message-meta';
      metaEl.textContent = `${data.model} • ${data.usage.completion_tokens || 0} tokens`;
      assistantBubble.appendChild(metaEl);

      tokenMetrics.textContent = `Tokens: Prompt: ${data.usage.prompt_tokens || 0} | Completion: ${data.usage.completion_tokens || 0}`;

      loadConversations();
    } catch (err) {
      assistantBubble.textContent = `Error: ${err.message}`;
      assistantBubble.style.color = '#fb7185';
    } finally {
      btnChatSend.disabled = false;
      scrollChatToBottom();
    }
  });

  // ── Memory Subsystem ──
  const memorySearchInput = document.getElementById('memory-search-input');
  const btnMemorySearch = document.getElementById('btn-memory-search');
  const memoryCardsContainer = document.getElementById('memory-cards-container');
  const memoryCategoryPills = document.getElementById('memory-category-pills');
  const btnOpenAddMemory = document.getElementById('btn-open-add-memory');
  const memoryModal = document.getElementById('memory-modal');
  const btnCloseMemoryModal = document.getElementById('btn-close-memory-modal');
  const btnCancelMemory = document.getElementById('btn-cancel-memory');
  const createMemoryForm = document.getElementById('create-memory-form');
  const memConfidence = document.getElementById('mem-confidence');
  const memConfVal = document.getElementById('mem-conf-val');

  memConfidence.addEventListener('input', () => {
    memConfVal.textContent = memConfidence.value;
  });

  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.searchMode = btn.dataset.mode;
      loadMemories();
    });
  });

  async function loadCategories() {
    try {
      const res = await fetch('/api/memories/categories');
      const data = await res.json();
      memoryCategoryPills.innerHTML = '<button class="cat-pill active" data-cat="">All</button>';
      (data.categories || []).forEach(cat => {
        const pill = document.createElement('button');
        pill.className = 'cat-pill';
        pill.dataset.cat = cat;
        pill.textContent = cat;
        pill.addEventListener('click', () => {
          document.querySelectorAll('.cat-pill').forEach(p => p.classList.remove('active'));
          pill.classList.add('active');
          state.selectedCategory = cat;
          loadMemories();
        });
        memoryCategoryPills.appendChild(pill);
      });
      // All pill click
      memoryCategoryPills.querySelector('.cat-pill').addEventListener('click', () => {
        document.querySelectorAll('.cat-pill').forEach(p => p.classList.remove('active'));
        memoryCategoryPills.querySelector('.cat-pill').classList.add('active');
        state.selectedCategory = '';
        loadMemories();
      });
    } catch (e) {
      console.error(e);
    }
  }

  async function loadMemories() {
    const q = memorySearchInput.value.trim();
    memoryCardsContainer.innerHTML = '<div class="loading-spinner">Searching memories...</div>';

    try {
      let url = `/api/memories?q=${encodeURIComponent(q)}&mode=${state.searchMode}`;
      if (state.selectedCategory) url += `&category=${encodeURIComponent(state.selectedCategory)}`;

      const res = await fetch(url);
      const data = await res.json();
      memoryCardsContainer.innerHTML = '';

      const items = data.results || [];
      if (items.length === 0) {
        memoryCardsContainer.innerHTML = '<div class="loading-spinner">No memories matched the search criteria</div>';
        return;
      }

      items.forEach(item => {
        const mem = item.memory ? item.memory : item;
        const score = item.score !== undefined ? item.score : item.similarity;

        const card = document.createElement('div');
        card.className = 'memory-card';

        card.innerHTML = `
          <div>
            <div class="memory-card-header">
              <span class="mem-cat-badge">${mem.category || 'general'}</span>
              <span class="mem-version-badge">v${mem.version} • conf ${mem.confidence}</span>
            </div>
            <div class="mem-content">${mem.content}</div>
          </div>
          <div class="mem-footer">
            <span>${score !== undefined ? `<span class="score-badge">Match: ${(score * 100).toFixed(0)}%</span>` : `ID: ${mem.id.slice(0, 8)}`}</span>
            <button class="btn btn-sm btn-danger btn-del-mem" data-id="${mem.id}">Delete</button>
          </div>
        `;

        card.querySelector('.btn-del-mem').addEventListener('click', async () => {
          if (confirm('Delete this memory record?')) {
            await fetch(`/api/memories/${mem.id}`, { method: 'DELETE' });
            loadMemories();
          }
        });

        memoryCardsContainer.appendChild(card);
      });
    } catch (e) {
      memoryCardsContainer.innerHTML = `<div class="loading-spinner">Error loading memories: ${e.message}</div>`;
    }
  }

  btnMemorySearch.addEventListener('click', loadMemories);
  memorySearchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') loadMemories();
  });

  // Modal open/close
  btnOpenAddMemory.addEventListener('click', () => { memoryModal.style.display = 'flex'; });
  btnCloseMemoryModal.addEventListener('click', () => { memoryModal.style.display = 'none'; });
  btnCancelMemory.addEventListener('click', () => { memoryModal.style.display = 'none'; });

  createMemoryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const content = document.getElementById('mem-content').value.trim();
    const category = document.getElementById('mem-category').value.trim();
    const source = document.getElementById('mem-source').value.trim();
    const confidence = parseFloat(memConfidence.value);
    const autoEmbed = document.getElementById('mem-auto-embed').checked;
    const consolidate = document.getElementById('mem-consolidate').checked;

    try {
      const res = await fetch('/api/memories', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content,
          category,
          source,
          confidence,
          auto_embed: autoEmbed,
          consolidate,
        }),
      });

      if (!res.ok) throw new Error(await res.text());

      memoryModal.style.display = 'none';
      createMemoryForm.reset();
      memConfidence.value = 1.0;
      memConfVal.textContent = '1.0';
      loadCategories();
      loadMemories();
    } catch (err) {
      alert(`Failed to save memory: ${err.message}`);
    }
  });

  // ── Knowledge Subsystem ──
  const knowledgeIndexForm = document.getElementById('knowledge-index-form');
  const docFilePath = document.getElementById('doc-file-path');
  const docTitle = document.getElementById('doc-title');
  const btnIndexDoc = document.getElementById('btn-index-doc');
  const indexResultAlert = document.getElementById('index-result-alert');
  const knowledgeSearchForm = document.getElementById('knowledge-search-form');
  const knowledgeQuery = document.getElementById('knowledge-query');
  const knowledgeSearchResults = document.getElementById('knowledge-search-results');
  const documentsTableBody = document.getElementById('documents-table-body');

  async function loadDocuments() {
    try {
      const res = await fetch('/api/knowledge/documents');
      const data = await res.json();
      documentsTableBody.innerHTML = '';

      if (!data.documents || data.documents.length === 0) {
        documentsTableBody.innerHTML = '<tr><td colspan="6" class="text-center">No indexed documents found</td></tr>';
        return;
      }

      data.documents.forEach(doc => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td><strong>${doc.title}</strong></td>
          <td style="font-family: var(--font-mono); font-size: 0.75rem;">${doc.source_path}</td>
          <td>${(doc.file_size_bytes / 1024).toFixed(1)} KB</td>
          <td><span class="mem-cat-badge">${doc.chunk_count} chunks</span></td>
          <td>${new Date(doc.updated_at).toLocaleString()}</td>
          <td><button class="btn btn-sm btn-danger btn-del-doc" data-id="${doc.id}">Delete</button></td>
        `;
        row.querySelector('.btn-del-doc').addEventListener('click', async () => {
          if (confirm(`Remove document '${doc.title}' and its vector embeddings?`)) {
            await fetch(`/api/knowledge/documents/${doc.id}`, { method: 'DELETE' });
            loadDocuments();
          }
        });
        documentsTableBody.appendChild(row);
      });
    } catch (e) {
      documentsTableBody.innerHTML = `<tr><td colspan="6" class="text-center">Failed to load documents: ${e.message}</td></tr>`;
    }
  }

  knowledgeIndexForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const filePath = docFilePath.value.trim();
    const title = docTitle.value.trim() || undefined;

    btnIndexDoc.disabled = true;
    indexResultAlert.style.display = 'block';
    indexResultAlert.textContent = 'Generating dense vector chunks via nomic-embed-text...';
    indexResultAlert.className = 'alert-box';

    try {
      const res = await fetch('/api/knowledge/index-file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path: filePath, title }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Indexing failed');

      indexResultAlert.className = 'alert-box success';
      indexResultAlert.textContent = `Indexed '${data.title}' into ${data.chunk_count} semantic vector chunks.`;
      docFilePath.value = '';
      docTitle.value = '';
      loadDocuments();
    } catch (err) {
      indexResultAlert.className = 'alert-box danger';
      indexResultAlert.textContent = err.message;
    } finally {
      btnIndexDoc.disabled = false;
    }
  });

  knowledgeSearchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = knowledgeQuery.value.trim();
    if (!q) return;

    knowledgeSearchResults.innerHTML = '<div class="loading-spinner">Searching semantic chunks...</div>';

    try {
      const res = await fetch(`/api/knowledge/search?q=${encodeURIComponent(q)}&limit=5`);
      const data = await res.json();
      knowledgeSearchResults.innerHTML = '';

      if (!data.results || data.results.length === 0) {
        knowledgeSearchResults.innerHTML = '<div class="loading-spinner">No relevant passages found</div>';
        return;
      }

      data.results.forEach(item => {
        const card = document.createElement('div');
        card.className = 'card mt-4';
        card.style.background = 'rgba(255,255,255,0.02)';
        card.innerHTML = `
          <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-weight: 600; font-size: 0.85rem; color: #38bdf8;">${item.document_title} (Chunk #${item.chunk_index})</span>
            <span class="score-badge">Similarity: ${(item.similarity * 100).toFixed(1)}%</span>
          </div>
          <div style="font-size: 0.88rem; line-height: 1.5; color: #e2e8f0; white-space: pre-wrap;">${item.content}</div>
        `;
        knowledgeSearchResults.appendChild(card);
      });
    } catch (e) {
      knowledgeSearchResults.innerHTML = `<div class="loading-spinner">Search error: ${e.message}</div>`;
    }
  });

  // ── Hardware & Models Subsystem ──
  const installedModelsGrid = document.getElementById('installed-models-grid');
  const btnRefreshHw = document.getElementById('btn-refresh-hw');

  async function loadHardwareAndModels() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      installedModelsGrid.innerHTML = '';

      if (!data.models || data.models.length === 0) {
        installedModelsGrid.innerHTML = '<div class="loading-spinner">No models found in local repository</div>';
        return;
      }

      // Populate select dropdown
      chatModelSelect.innerHTML = '';
      data.models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = `${m.name} (${m.parameter_count || (m.size_bytes / 1e9).toFixed(1) + 'GB'})`;
        if (m.name === state.selectedModel) opt.selected = true;
        chatModelSelect.appendChild(opt);

        // Populate hardware tab card
        const card = document.createElement('div');
        card.className = 'metric-card';
        card.innerHTML = `
          <div class="metric-header">
            <span class="metric-label">${m.family || 'LLM'}</span>
            <span class="metric-badge">${m.quantization || 'Q4_K'}</span>
          </div>
          <div class="metric-value" style="font-size: 1.05rem;">${m.name}</div>
          <div class="metric-sub">${(m.size_bytes / 1e9).toFixed(2)} GB • GPU Accelerated</div>
        `;
        installedModelsGrid.appendChild(card);
      });

      chatModelSelect.addEventListener('change', () => {
        state.selectedModel = chatModelSelect.value;
        activeModelTag.textContent = state.selectedModel;
      });
    } catch (e) {
      installedModelsGrid.innerHTML = `<div class="loading-spinner">Error: ${e.message}</div>`;
    }
  }

  btnRefreshHw.addEventListener('click', loadHardwareAndModels);

  // ── Backups Subsystem ──
  const btnCreateBackup = document.getElementById('btn-create-backup');
  const backupsTableBody = document.getElementById('backups-table-body');

  async function loadBackups() {
    try {
      const res = await fetch('/api/backups');
      const data = await res.json();
      backupsTableBody.innerHTML = '';

      if (!data.backups || data.backups.length === 0) {
        backupsTableBody.innerHTML = '<tr><td colspan="5" class="text-center">No backup snapshots found</td></tr>';
        return;
      }

      data.backups.forEach(b => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td><strong>${b.filename}</strong></td>
          <td>${(b.size_bytes / 1024).toFixed(1)} KB</td>
          <td>${new Date(b.created_at).toLocaleString()}</td>
          <td><span class="status-indicator online" style="display:inline-flex;">Verified OK</span></td>
          <td style="font-family: var(--font-mono); font-size: 0.75rem;">${b.path}</td>
        `;
        backupsTableBody.appendChild(row);
      });
    } catch (e) {
      backupsTableBody.innerHTML = `<tr><td colspan="5" class="text-center">Error loading backups: ${e.message}</td></tr>`;
    }
  }

  btnCreateBackup.addEventListener('click', async () => {
    btnCreateBackup.disabled = true;
    try {
      const res = await fetch('/api/backups', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        loadBackups();
      } else {
        alert('Backup creation failed');
      }
    } catch (e) {
      alert(`Error creating backup: ${e.message}`);
    } finally {
      btnCreateBackup.disabled = false;
    }
  });

  // ── Tools & Tasks Subsystem ──
  const toolsGrid = document.getElementById('tools-grid');
  const tasksTableBody = document.getElementById('tasks-table-body');
  const btnRefreshTasks = document.getElementById('btn-refresh-tasks');

  async function loadTools() {
    try {
      const res = await fetch('/api/tools');
      const data = await res.json();
      toolsGrid.innerHTML = '';

      if (!data.tools || data.tools.length === 0) {
        toolsGrid.innerHTML = '<p class="text-muted">No tools registered.</p>';
        return;
      }

      data.tools.forEach(tool => {
        const card = document.createElement('div');
        card.className = 'tool-card card';
        const isReadOnly = tool.permission === 'read_only';
        card.innerHTML = `
          <div class="card-header-flex">
            <h4>${tool.name}</h4>
            <span class="badge ${isReadOnly ? 'badge-info' : 'badge-warning'}">${tool.permission}</span>
          </div>
          <p class="tool-desc mt-2" style="font-size: 0.85rem; color: var(--text-muted);">${tool.description}</p>
          <div class="tool-params mt-3">
            <span style="font-size: 0.75rem; text-transform: uppercase; color: var(--text-dim); font-weight: 600;">Parameters:</span>
            <code style="display: block; font-size: 0.75rem; color: var(--accent-light); margin-top: 4px;">
              ${tool.parameters.map(p => `${p.name}${p.required ? '*' : ''}`).join(', ') || 'none'}
            </code>
          </div>
        `;
        toolsGrid.appendChild(card);
      });
    } catch (e) {
      toolsGrid.innerHTML = `<p class="text-muted">Error loading tools: ${e.message}</p>`;
    }
  }

  async function loadTasks() {
    try {
      const res = await fetch('/api/tasks');
      const data = await res.json();
      tasksTableBody.innerHTML = '';

      if (!data.tasks || data.tasks.length === 0) {
        tasksTableBody.innerHTML = '<tr><td colspan="5" class="text-center">No task traces recorded</td></tr>';
        return;
      }

      data.tasks.forEach(task => {
        const row = document.createElement('tr');
        const isCompleted = task.status === 'completed';
        const isFailed = task.status === 'failed';
        const badgeClass = isCompleted ? 'online' : (isFailed ? 'offline' : 'pending');

        row.innerHTML = `
          <td><code style="font-size: 0.75rem;">${task.id.substring(0, 8)}...</code></td>
          <td><strong>${task.goal}</strong></td>
          <td><span class="status-indicator ${badgeClass}">${task.status}</span></td>
          <td>${task.current_step} / ${task.total_steps}</td>
          <td>${new Date(task.created_at).toLocaleTimeString()}</td>
        `;
        tasksTableBody.appendChild(row);
      });
    } catch (e) {
      tasksTableBody.innerHTML = `<tr><td colspan="5" class="text-center">Error loading tasks: ${e.message}</td></tr>`;
    }
  }

  if (btnRefreshTasks) {
    btnRefreshTasks.addEventListener('click', loadTasks);
  }

  // ── Evaluation & Benchmarking Subsystem ──
  const benchmarksGrid = document.getElementById('benchmarks-grid');
  const evalRunsTableBody = document.getElementById('eval-runs-table-body');
  const btnRefreshEvals = document.getElementById('btn-refresh-evals');
  const evalBenchmarksCount = document.getElementById('eval-benchmarks-count');
  const evalLatestScore = document.getElementById('eval-latest-score');
  const evalLatestBadge = document.getElementById('eval-latest-badge');
  const evalLatestSub = document.getElementById('eval-latest-sub');
  const evalRegressionStatus = document.getElementById('eval-regression-status');
  const evalAvgLatency = document.getElementById('eval-avg-latency');
  const evalDetailsCard = document.getElementById('eval-details-card');
  const evalDetailsTitle = document.getElementById('eval-details-title');
  const evalDetailsContent = document.getElementById('eval-details-content');
  const btnCloseEvalDetails = document.getElementById('btn-close-eval-details');

  if (btnCloseEvalDetails) {
    btnCloseEvalDetails.addEventListener('click', () => {
      evalDetailsCard.style.display = 'none';
    });
  }

  if (btnRefreshEvals) {
    btnRefreshEvals.addEventListener('click', () => {
      loadBenchmarks();
      loadEvalRuns();
    });
  }

  async function loadBenchmarks() {
    try {
      const res = await fetch('/api/evaluation/benchmarks');
      const data = await res.json();
      if (!benchmarksGrid) return;
      benchmarksGrid.innerHTML = '';

      if (!data.benchmarks || data.benchmarks.length === 0) {
        benchmarksGrid.innerHTML = '<p class="text-muted">No benchmark suites registered</p>';
        return;
      }

      if (evalBenchmarksCount) evalBenchmarksCount.textContent = `${data.benchmarks.length} Available`;

      data.benchmarks.forEach(bm => {
        const card = document.createElement('div');
        card.className = 'tool-card';
        card.innerHTML = `
          <div class="tool-card-header">
            <h4>${bm.name}</h4>
            <span class="tool-badge badge-read">${bm.category.toUpperCase()}</span>
          </div>
          <p class="tool-desc">${bm.description}</p>
          <div class="tool-meta mt-2" style="font-size: 0.8rem; color: var(--text-muted);">
            <span>📋 <strong>${bm.case_count}</strong> test cases</span> • 
            <span>Rubrics: Exact, Contains, Regex</span>
          </div>
          <div class="mt-3">
            <button class="btn btn-sm btn-primary btn-run-bm" data-bm-id="${bm.id}" data-bm-name="${bm.name}">
              ▶️ Execute Suite
            </button>
          </div>
        `;

        card.querySelector('.btn-run-bm').addEventListener('click', () => {
          runBenchmarkSuite(bm.id, bm.name);
        });

        benchmarksGrid.appendChild(card);
      });
    } catch (e) {
      if (benchmarksGrid) benchmarksGrid.innerHTML = `<p class="text-danger">Failed to load benchmarks: ${e.message}</p>`;
    }
  }

  async function runBenchmarkSuite(benchmarkId, benchmarkName) {
    if (!confirm(`Execute evaluation suite "${benchmarkName}" against model "${state.selectedModel}"?`)) return;

    try {
      if (evalDetailsCard) {
        evalDetailsCard.style.display = 'block';
        evalDetailsTitle.textContent = `Running Suite: ${benchmarkName}...`;
        evalDetailsContent.innerHTML = `
          <div class="loading-spinner">
            Evaluating test cases against ${state.selectedModel} via local GPU inference. Please wait...
          </div>
        `;
      }

      const res = await fetch('/api/evaluation/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          benchmark_id: benchmarkId,
          model: state.selectedModel,
          temperature: 0.1,
          max_tokens: 250,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Evaluation run failed');
      }

      const run = await res.json();
      loadEvalRuns();
      viewRunReport(run.id);
    } catch (e) {
      alert(`Error running benchmark: ${e.message}`);
      if (evalDetailsCard) evalDetailsCard.style.display = 'none';
    }
  }

  async function loadEvalRuns() {
    try {
      const res = await fetch('/api/evaluation/runs');
      const data = await res.json();
      if (!evalRunsTableBody) return;
      evalRunsTableBody.innerHTML = '';

      if (!data.runs || data.runs.length === 0) {
        evalRunsTableBody.innerHTML = '<tr><td colspan="8" class="text-center">No benchmark evaluation runs recorded yet</td></tr>';
        return;
      }

      // Update top metrics from latest run
      const latest = data.runs[0];
      if (evalLatestScore) {
        evalLatestScore.textContent = `${latest.score.toFixed(1)}%`;
        evalLatestBadge.textContent = latest.score >= 80 ? 'Optimal' : (latest.score >= 50 ? 'Acceptable' : 'Deficient');
        evalLatestSub.textContent = `${latest.passed_cases}/${latest.total_cases} passed • ${latest.model_id}`;
      }

      // Compute average latency
      const totalDur = data.runs.reduce((acc, r) => acc + (r.duration_ms || 0), 0);
      const totalCases = data.runs.reduce((acc, r) => acc + (r.total_cases || 0), 0);
      if (evalAvgLatency && totalCases > 0) {
        evalAvgLatency.textContent = `${Math.round(totalDur / totalCases)} ms/case`;
      }

      data.runs.forEach(run => {
        const row = document.createElement('tr');
        const scoreClass = run.score >= 80 ? 'text-success' : (run.score >= 50 ? 'text-warning' : 'text-danger');

        row.innerHTML = `
          <td><code style="font-size: 0.75rem;">${run.id.substring(0, 8)}...</code></td>
          <td><strong>${run.benchmark_id}</strong></td>
          <td><span class="chat-model-tag">${run.model_id}</span></td>
          <td><strong class="${scoreClass}" style="font-size: 1rem;">${run.score.toFixed(1)}%</strong></td>
          <td>${run.passed_cases} / ${run.total_cases}</td>
          <td>${(run.duration_ms / 1000).toFixed(2)}s</td>
          <td>${new Date(run.started_at).toLocaleTimeString()}</td>
          <td>
            <button class="btn btn-sm btn-secondary btn-view-report" data-run-id="${run.id}">📄 Report</button>
            <button class="btn btn-sm btn-primary btn-compare-run" data-run-id="${run.id}">⚖️ Compare</button>
          </td>
        `;

        row.querySelector('.btn-view-report').addEventListener('click', () => viewRunReport(run.id));
        row.querySelector('.btn-compare-run').addEventListener('click', () => compareRunBaseline(run.id));

        evalRunsTableBody.appendChild(row);
      });
    } catch (e) {
      if (evalRunsTableBody) evalRunsTableBody.innerHTML = `<tr><td colspan="8" class="text-center">Error loading runs: ${e.message}</td></tr>`;
    }
  }

  async function viewRunReport(runId) {
    try {
      if (!evalDetailsCard) return;
      evalDetailsCard.style.display = 'block';
      evalDetailsTitle.textContent = `Evaluation Run Report: ${runId.substring(0, 8)}...`;
      evalDetailsContent.innerHTML = '<div class="loading-spinner">Generating structured report...</div>';

      const res = await fetch(`/api/evaluation/runs/${runId}/report`);
      if (!res.ok) throw new Error('Failed to fetch report');
      const data = await res.json();
      const run = data.json;

      const failSummary = run.metadata && run.metadata.failure_summary ? run.metadata.failure_summary : {};

      let html = `
        <div class="metrics-grid mb-4">
          <div class="metric-card">
            <div class="metric-label">Benchmark Suite</div>
            <div class="metric-value">${run.benchmark_id}</div>
            <div class="metric-sub">${run.model_id}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Pass Rate</div>
            <div class="metric-value ${run.score >= 80 ? 'text-success' : 'text-danger'}">${run.score.toFixed(1)}%</div>
            <div class="metric-sub">${run.passed_cases} passed, ${run.failed_cases} failed</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Dominant Failure Mode</div>
            <div class="metric-value" style="font-size: 1.1rem;"><code>${failSummary.dominant_failure || 'none'}</code></div>
            <div class="metric-sub">Total Failures: ${failSummary.total_failures || 0}</div>
          </div>
        </div>

        <div class="card mb-4" style="border-left: 3px solid var(--accent-cyan); background: rgba(6, 182, 212, 0.05);">
          <h4>💡 Actionable Recommendation</h4>
          <p class="mt-1">${failSummary.recommendation || 'All test cases passed.'}</p>
        </div>

        <h4>Detailed Case Outcomes</h4>
        <div class="table-responsive mt-2">
          <table class="data-table">
            <thead>
              <tr>
                <th>Case ID</th>
                <th>Status</th>
                <th>Score</th>
                <th>Latency</th>
                <th>Failure Category</th>
                <th>Output / Error</th>
              </tr>
            </thead>
            <tbody>
      `;

      run.results.forEach(res => {
        const passIcon = res.passed ? '✅ Pass' : '❌ Fail';
        const passClass = res.passed ? 'text-success' : 'text-danger';
        const snippet = (res.actual_output || res.error || '').substring(0, 100);

        html += `
          <tr>
            <td><code>${res.case_id}</code></td>
            <td><strong class="${passClass}">${passIcon}</strong></td>
            <td>${res.score.toFixed(2)}</td>
            <td>${Math.round(res.latency_ms)} ms</td>
            <td><span class="tool-badge ${res.passed ? 'badge-read' : 'badge-danger'}">${res.failure_category}</span></td>
            <td><code style="font-size: 0.75rem;">${snippet}</code></td>
          </tr>
        `;
      });

      html += `
            </tbody>
          </table>
        </div>
      `;

      evalDetailsContent.innerHTML = html;
      evalDetailsCard.scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
      if (evalDetailsContent) evalDetailsContent.innerHTML = `<p class="text-danger">Failed to display report: ${e.message}</p>`;
    }
  }

  async function compareRunBaseline(candidateRunId) {
    try {
      if (!evalDetailsCard) return;
      evalDetailsCard.style.display = 'block';
      evalDetailsTitle.textContent = `Regression Analysis: ${candidateRunId.substring(0, 8)}...`;
      evalDetailsContent.innerHTML = '<div class="loading-spinner">Performing regression analysis against baseline...</div>';

      const res = await fetch('/api/evaluation/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidate_run_id: candidateRunId }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to compare runs');
      }

      const data = await res.json();
      const comp = data.comparison;

      const verdictBadgeClass = comp.verdict === 'PASSED' ? 'badge-read' : (comp.verdict === 'IMPROVED' ? 'badge-read' : 'badge-danger');
      const scoreDeltaSign = comp.score_delta > 0 ? '+' : '';

      let html = `
        <div class="metrics-grid mb-4">
          <div class="metric-card">
            <div class="metric-label">Regression Verdict</div>
            <div class="metric-value"><span class="tool-badge ${verdictBadgeClass}" style="font-size: 1rem;">${comp.verdict}</span></div>
            <div class="metric-sub">${comp.regressed ? 'Action Required' : 'Verified Stable'}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Score Delta</div>
            <div class="metric-value ${comp.score_delta >= 0 ? 'text-success' : 'text-danger'}">${scoreDeltaSign}${comp.score_delta.toFixed(1)}%</div>
            <div class="metric-sub">Base: ${comp.baseline_score.toFixed(1)}% → Cand: ${comp.candidate_score.toFixed(1)}%</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Latency Impact</div>
            <div class="metric-value">${comp.latency_change_percent > 0 ? '+' : ''}${comp.latency_change_percent.toFixed(1)}%</div>
            <div class="metric-sub">${(comp.baseline_duration_ms / 1000).toFixed(2)}s → ${(comp.candidate_duration_ms / 1000).toFixed(2)}s</div>
          </div>
        </div>

        <div class="card mb-4" style="border-left: 3px solid ${comp.regressed ? 'var(--accent-rose)' : 'var(--accent-emerald)'}; background: rgba(16, 185, 129, 0.05);">
          <h4>📋 Comparison Summary</h4>
          <p class="mt-1">${comp.summary}</p>
        </div>
      `;

      if (comp.regressed_cases && comp.regressed_cases.length > 0) {
        html += `
          <h4 class="text-danger">🚨 Regressed Test Cases (${comp.regressed_cases.length})</h4>
          <div class="table-responsive mt-2 mb-4">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Case ID</th>
                  <th>Base Score</th>
                  <th>Cand Score</th>
                  <th>Failure Category</th>
                  <th>Candidate Output / Error</th>
                </tr>
              </thead>
              <tbody>
        `;
        comp.regressed_cases.forEach(rc => {
          html += `
            <tr>
              <td><code>${rc.case_id}</code></td>
              <td>${rc.baseline_score.toFixed(2)}</td>
              <td><strong class="text-danger">${rc.candidate_score.toFixed(2)}</strong></td>
              <td><span class="tool-badge badge-danger">${rc.failure_category}</span></td>
              <td><code>${(rc.actual_output || rc.error || '').substring(0, 100)}</code></td>
            </tr>
          `;
        });
        html += `</tbody></table></div>`;
      }

      evalDetailsContent.innerHTML = html;
      evalDetailsCard.scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
      if (evalDetailsContent) evalDetailsContent.innerHTML = `<p class="text-danger">Comparison failed: ${e.message}</p>`;
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  // ── Phase 5: Datasets & Fine-Tuning Controller ──
  // ═══════════════════════════════════════════════════════════════════════════════

  const trainingDatasetsCount = document.getElementById('training-datasets-count');
  const trainingSamplesCount = document.getElementById('training-samples-count');
  const trainingTokensSub = document.getElementById('training-tokens-sub');
  const trainingExperimentsCount = document.getElementById('training-experiments-count');
  const trainingExpSub = document.getElementById('training-exp-sub');
  const datasetsTableBody = document.getElementById('datasets-table-body');
  const experimentsTableBody = document.getElementById('experiments-table-body');
  const trainingDetailsCard = document.getElementById('training-details-card');
  const trainingDetailsTitle = document.getElementById('training-details-title');
  const trainingDetailsContent = document.getElementById('training-details-content');
  const btnCloseTrainingDetails = document.getElementById('btn-close-training-details');

  const datasetModal = document.getElementById('dataset-modal');
  const btnOpenIngestDataset = document.getElementById('btn-open-ingest-dataset');
  const btnCloseDatasetModal = document.getElementById('btn-close-dataset-modal');
  const btnCancelDataset = document.getElementById('btn-cancel-dataset');
  const ingestDatasetForm = document.getElementById('ingest-dataset-form');

  const experimentModal = document.getElementById('experiment-modal');
  const btnOpenNewExperiment = document.getElementById('btn-open-new-experiment');
  const btnCloseExperimentModal = document.getElementById('btn-close-experiment-modal');
  const btnCancelExperiment = document.getElementById('btn-cancel-experiment');
  const launchExperimentForm = document.getElementById('launch-experiment-form');
  const expBaseModelSelect = document.getElementById('exp-base-model');
  const expDatasetSelect = document.getElementById('exp-dataset-select');

  const btnRefreshTraining = document.getElementById('btn-refresh-training');
  const btnRefreshDatasetsTable = document.getElementById('btn-refresh-datasets-table');
  const btnRefreshExperimentsTable = document.getElementById('btn-refresh-experiments-table');

  if (btnCloseTrainingDetails) {
    btnCloseTrainingDetails.addEventListener('click', () => {
      trainingDetailsCard.style.display = 'none';
    });
  }

  // Modal open/close bindings
  if (btnOpenIngestDataset) {
    btnOpenIngestDataset.addEventListener('click', () => {
      datasetModal.style.display = 'flex';
    });
  }
  const closeDatasetModal = () => { if (datasetModal) datasetModal.style.display = 'none'; };
  if (btnCloseDatasetModal) btnCloseDatasetModal.addEventListener('click', closeDatasetModal);
  if (btnCancelDataset) btnCancelDataset.addEventListener('click', closeDatasetModal);

  if (btnOpenNewExperiment) {
    btnOpenNewExperiment.addEventListener('click', () => {
      // Populate models in experiment select
      if (expBaseModelSelect) {
        expBaseModelSelect.innerHTML = '';
        state.models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.name;
          opt.textContent = `${m.name} (${m.size_gb.toFixed(1)} GB)`;
          if (m.name.includes('qwen2.5')) opt.selected = true;
          expBaseModelSelect.appendChild(opt);
        });
      }
      experimentModal.style.display = 'flex';
    });
  }
  const closeExperimentModal = () => { if (experimentModal) experimentModal.style.display = 'none'; };
  if (btnCloseExperimentModal) btnCloseExperimentModal.addEventListener('click', closeExperimentModal);
  if (btnCancelExperiment) btnCancelExperiment.addEventListener('click', closeExperimentModal);

  if (btnRefreshTraining) btnRefreshTraining.addEventListener('click', loadTrainingData);
  if (btnRefreshDatasetsTable) btnRefreshDatasetsTable.addEventListener('click', loadDatasets);
  if (btnRefreshExperimentsTable) btnRefreshExperimentsTable.addEventListener('click', loadExperiments);

  // Ingest Dataset submit
  if (ingestDatasetForm) {
    ingestDatasetForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const name = document.getElementById('ds-name').value.trim();
      const format = document.getElementById('ds-format').value;
      const description = document.getElementById('ds-description').value.trim();
      const rawContent = document.getElementById('ds-content').value.trim();

      let parsedSamples = [];
      try {
        if (rawContent.startsWith('[')) {
          parsedSamples = JSON.parse(rawContent);
        } else {
          // JSONL lines
          parsedSamples = rawContent.split('\n')
            .map(l => l.trim())
            .filter(l => l.length > 0)
            .map(l => JSON.parse(l));
        }
      } catch (err) {
        alert('Invalid JSON/JSONL format in samples: ' + err.message);
        return;
      }

      try {
        const res = await fetch('/api/training/datasets/create', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name,
            description,
            format,
            samples: parsedSamples,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        closeDatasetModal();
        ingestDatasetForm.reset();
        await loadDatasets();
      } catch (err) {
        alert('Failed to register dataset: ' + err.message);
      }
    });
  }

  // Launch Experiment submit
  if (launchExperimentForm) {
    launchExperimentForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const name = document.getElementById('exp-name').value.trim();
      const baseModel = expBaseModelSelect.value;
      const datasetId = expDatasetSelect.value;
      const loraR = parseInt(document.getElementById('exp-lora-r').value, 10);
      const loraAlpha = parseInt(document.getElementById('exp-lora-alpha').value, 10);
      const epochs = parseInt(document.getElementById('exp-epochs').value, 10);
      const batchSize = parseInt(document.getElementById('exp-batch-size').value, 10);
      const quantization = document.getElementById('exp-quantization').value;

      try {
        const res = await fetch('/api/training/experiments/create', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name,
            base_model: baseModel,
            dataset_id: datasetId,
            hyperparameters: {
              lora_r: loraR,
              lora_alpha: loraAlpha,
              epochs,
              batch_size: batchSize,
              quantization,
            },
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        closeExperimentModal();
        launchExperimentForm.reset();
        await loadExperiments();
      } catch (err) {
        alert('Failed to create experiment: ' + err.message);
      }
    });
  }

  async function loadTrainingData() {
    await Promise.all([loadDatasets(), loadExperiments()]);
  }

  async function loadDatasets() {
    if (!datasetsTableBody) return;
    try {
      const res = await fetch('/api/training/datasets');
      const data = await res.json();
      const datasets = data.datasets || [];

      if (trainingDatasetsCount) trainingDatasetsCount.textContent = datasets.length;
      let totalSamples = 0;
      let totalTokens = 0;
      datasets.forEach(d => {
        totalSamples += (d.sample_count || 0);
        totalTokens += (d.total_tokens || 0);
      });
      if (trainingSamplesCount) trainingSamplesCount.textContent = totalSamples.toLocaleString();
      if (trainingTokensSub) trainingTokensSub.textContent = `~${totalTokens.toLocaleString()} tokens`;

      // Update experiment modal select options
      if (expDatasetSelect) {
        expDatasetSelect.innerHTML = '';
        datasets.forEach(d => {
          const opt = document.createElement('option');
          opt.value = d.id;
          opt.textContent = `${d.name} (${d.sample_count} samples • ${d.format})`;
          expDatasetSelect.appendChild(opt);
        });
      }

      if (datasets.length === 0) {
        datasetsTableBody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No training datasets registered yet. Click "+ Ingest Dataset" to register one.</td></tr>`;
        return;
      }

      datasetsTableBody.innerHTML = datasets.map(d => {
        const created = new Date(d.created_at).toLocaleString();
        return `
          <tr>
            <td>
              <strong>${d.name}</strong>
              <div class="text-muted text-sm">${d.description || 'No description'}</div>
            </td>
            <td><span class="tool-badge badge-read">${d.format}</span></td>
            <td><code>${d.version}</code></td>
            <td><strong>${(d.sample_count || 0).toLocaleString()}</strong></td>
            <td>${(d.total_tokens || 0).toLocaleString()}</td>
            <td class="text-sm">${created}</td>
            <td>
              <div class="btn-group">
                <button class="btn btn-secondary btn-sm btn-preprocess-ds" data-id="${d.id}" title="Run deduplication, syntax checking, and eval contamination guard">🛡️ Clean</button>
              </div>
            </td>
          </tr>
        `;
      }).join('');

      // Bind preprocess buttons
      document.querySelectorAll('.btn-preprocess-ds').forEach(btn => {
        btn.addEventListener('click', async () => {
          const dsId = btn.dataset.id;
          btn.disabled = true;
          btn.textContent = 'Cleaning...';
          try {
            const prepRes = await fetch(`/api/training/datasets/${dsId}/preprocess`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                min_tokens: 3,
                max_tokens: 4096,
                deduplicate: true,
                validate_syntax: true,
                check_contamination: true,
              }),
            });
            if (!prepRes.ok) throw new Error(await prepRes.text());
            const prepData = await prepRes.json();
            const rep = prepData.report;
            alert(`Preprocessing complete!\n• Initial: ${rep.initial_count} samples\n• Retained: ${rep.final_count} samples (${rep.retained_percentage}%)\n• Duplicates removed: ${rep.duplicates_removed}\n• Syntax errors filtered: ${rep.syntax_errors_filtered}\n• Benchmark contaminated removed: ${rep.contaminated_removed}`);
            await loadDatasets();
          } catch (err) {
            alert('Preprocessing error: ' + err.message);
          } finally {
            btn.disabled = false;
            btn.textContent = '🛡️ Clean';
          }
        });
      });
    } catch (e) {
      if (datasetsTableBody) datasetsTableBody.innerHTML = `<tr><td colspan="7" class="text-danger text-center">Failed to load datasets: ${e.message}</td></tr>`;
    }
  }

  async function loadExperiments() {
    if (!experimentsTableBody) return;
    try {
      const res = await fetch('/api/training/experiments');
      const data = await res.json();
      const experiments = data.experiments || [];

      if (trainingExperimentsCount) trainingExperimentsCount.textContent = experiments.length;
      const completedCount = experiments.filter(e => e.status === 'completed').length;
      if (trainingExpSub) trainingExpSub.textContent = `${completedCount} completed`;

      if (experiments.length === 0) {
        experimentsTableBody.innerHTML = `<tr><td colspan="8" class="text-center text-muted">No fine-tuning experiments found. Click "⚡ Launch Experiment" to create one.</td></tr>`;
        return;
      }

      experimentsTableBody.innerHTML = experiments.map(exp => {
        const hp = exp.hyperparameters || {};
        const isCompleted = exp.status === 'completed';
        const isTraining = exp.status === 'training';
        const isPending = exp.status === 'pending';

        let statusBadge = '<span class="status-indicator pending">Pending</span>';
        if (isCompleted) statusBadge = '<span class="status-indicator online">Completed</span>';
        if (isTraining) statusBadge = '<span class="status-indicator warning">Training</span>';
        if (exp.status === 'failed') statusBadge = '<span class="status-indicator offline">Failed</span>';

        const lossDisplay = exp.training_loss != null ? exp.training_loss.toFixed(4) : '—';
        const durationDisplay = exp.duration_ms ? `${(exp.duration_ms / 1000).toFixed(1)}s` : '—';

        return `
          <tr>
            <td>
              <strong>${exp.name}</strong>
              <div class="text-muted text-sm">r=${hp.lora_r || 16} • ${hp.quantization || '4bit'}</div>
            </td>
            <td><code>${exp.base_model}</code></td>
            <td><span class="text-sm text-dim">${exp.dataset_id}</span></td>
            <td>${statusBadge}</td>
            <td><strong>${exp.current_step}</strong> / ${exp.total_steps} (Ep ${exp.current_epoch}/${exp.total_epochs})</td>
            <td><code class="text-success">${lossDisplay}</code></td>
            <td>${durationDisplay}</td>
            <td>
              <div class="btn-group">
                ${isPending ? `<button class="btn btn-primary btn-sm btn-start-exp" data-id="${exp.id}">▶ Start</button>` : ''}
                <button class="btn btn-secondary btn-sm btn-view-exp" data-id="${exp.id}">🔍 Details</button>
                ${isCompleted ? `<button class="btn btn-secondary btn-sm btn-rollback-exp" data-id="${exp.id}" data-base="${exp.base_model}">↩ Rollback</button>` : ''}
              </div>
            </td>
          </tr>
        `;
      }).join('');

      // Bind Start experiment buttons
      document.querySelectorAll('.btn-start-exp').forEach(btn => {
        btn.addEventListener('click', async () => {
          const expId = btn.dataset.id;
          btn.disabled = true;
          btn.textContent = 'Training...';
          try {
            const startRes = await fetch(`/api/training/experiments/${expId}/start?step_delay_s=0.02`, {
              method: 'POST',
            });
            if (!startRes.ok) throw new Error(await startRes.text());
            await loadExperiments();
            await viewExperimentDetails(expId);
          } catch (err) {
            alert('Training failed: ' + err.message);
          } finally {
            btn.disabled = false;
            btn.textContent = '▶ Start';
          }
        });
      });

      // Bind View Details buttons
      document.querySelectorAll('.btn-view-exp').forEach(btn => {
        btn.addEventListener('click', () => {
          viewExperimentDetails(btn.dataset.id);
        });
      });

      // Bind Rollback buttons
      document.querySelectorAll('.btn-rollback-exp').forEach(btn => {
        btn.addEventListener('click', async () => {
          const expId = btn.dataset.id;
          const baseModel = btn.dataset.base || 'qwen2.5:7b';
          if (!confirm(`Rollback candidate model to baseline '${baseModel}'?`)) return;
          try {
            const rbRes = await fetch(`/api/training/experiments/${expId}/rollback`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ baseline_model: baseModel }),
            });
            if (!rbRes.ok) throw new Error(await rbRes.text());
            const rbData = await rbRes.json();
            alert(`Rollback successful! Active model reverted to baseline '${rbData.active_baseline}'.`);
            await loadExperiments();
          } catch (err) {
            alert('Rollback failed: ' + err.message);
          }
        });
      });
    } catch (e) {
      if (experimentsTableBody) experimentsTableBody.innerHTML = `<tr><td colspan="8" class="text-danger text-center">Failed to load experiments: ${e.message}</td></tr>`;
    }
  }

  async function viewExperimentDetails(experimentId) {
    if (!trainingDetailsCard || !trainingDetailsContent) return;
    trainingDetailsCard.style.display = 'block';
    trainingDetailsContent.innerHTML = `<div class="loading-spinner">Loading experiment trajectory & checkpoints...</div>`;

    try {
      const [expRes, chkRes] = await Promise.all([
        fetch(`/api/training/experiments/${experimentId}`),
        fetch(`/api/training/experiments/${experimentId}/checkpoints`),
      ]);
      const exp = await expRes.json();
      const chkData = await chkRes.json();
      const checkpoints = chkData.checkpoints || [];
      const bestChkId = chkData.best_checkpoint_id;

      trainingDetailsTitle.textContent = `Experiment: ${exp.name} (${exp.id})`;

      let html = `
        <div class="metrics-grid mb-4">
          <div class="metric-card">
            <div class="metric-label">Status</div>
            <div class="metric-value">${exp.status.toUpperCase()}</div>
            <div class="metric-sub">Base Model: ${exp.base_model}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Final Loss</div>
            <div class="metric-value text-success">${exp.training_loss != null ? exp.training_loss.toFixed(4) : '—'}</div>
            <div class="metric-sub">Eval Loss: ${exp.eval_loss != null ? exp.eval_loss.toFixed(4) : '—'}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Progress</div>
            <div class="metric-value">${exp.current_step} / ${exp.total_steps}</div>
            <div class="metric-sub">Epoch ${exp.current_epoch} of ${exp.total_epochs}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Checkpoints Saved</div>
            <div class="metric-value">${checkpoints.length}</div>
            <div class="metric-sub">${bestChkId ? 'Best checkpoint identified' : 'In training'}</div>
          </div>
        </div>

        <h4>📈 Training Loss Trajectory (Last 10 Metric Steps)</h4>
        <div class="table-responsive mt-2 mb-4">
          <table class="data-table">
            <thead>
              <tr>
                <th>Step</th>
                <th>Epoch</th>
                <th>Train Loss</th>
                <th>Eval Loss</th>
                <th>Learning Rate</th>
                <th>Tokens/sec</th>
                <th>VRAM Allocated</th>
              </tr>
            </thead>
            <tbody>
      `;

      const recentMetrics = (exp.metrics || []).slice(-10);
      if (recentMetrics.length === 0) {
        html += `<tr><td colspan="7" class="text-center text-muted">No step metrics recorded yet.</td></tr>`;
      } else {
        recentMetrics.forEach(m => {
          html += `
            <tr>
              <td><strong>#${m.step}</strong></td>
              <td>${m.epoch.toFixed(2)}</td>
              <td><code>${m.loss.toFixed(4)}</code></td>
              <td><code>${m.eval_loss != null ? m.eval_loss.toFixed(4) : '—'}</code></td>
              <td>${m.learning_rate.toExponential(2)}</td>
              <td>${m.tokens_per_sec.toFixed(1)}</td>
              <td>${(m.vram_mb / 1024).toFixed(2)} GB</td>
            </tr>
          `;
        });
      }
      html += `</tbody></table></div>`;

      // Checkpoints table
      html += `
        <h4>💾 Emitted Checkpoints &amp; Ollama Deployment</h4>
        <div class="table-responsive mt-2 mb-3">
          <table class="data-table">
            <thead>
              <tr>
                <th>Checkpoint ID</th>
                <th>Step</th>
                <th>Epoch</th>
                <th>Loss</th>
                <th>Best</th>
                <th>Deployed Name</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
      `;

      if (checkpoints.length === 0) {
        html += `<tr><td colspan="7" class="text-center text-muted">No checkpoints emitted yet.</td></tr>`;
      } else {
        checkpoints.forEach(chk => {
          const isBest = chk.id === bestChkId || chk.is_best;
          html += `
            <tr>
              <td><code>${chk.id}</code></td>
              <td><strong>Step ${chk.step}</strong></td>
              <td>Ep ${chk.epoch.toFixed(1)}</td>
              <td><code>${chk.training_loss.toFixed(4)}</code></td>
              <td>${isBest ? '<span class="tool-badge badge-read">★ Best</span>' : '—'}</td>
              <td><code>${chk.deployed_name || 'Not deployed'}</code></td>
              <td>
                <button class="btn btn-primary btn-sm btn-deploy-chk" data-id="${chk.id}" data-model="${exp.base_model}">🚀 Deploy to Ollama</button>
              </td>
            </tr>
          `;
        });
      }
      html += `</tbody></table></div>`;

      trainingDetailsContent.innerHTML = html;
      trainingDetailsCard.scrollIntoView({ behavior: 'smooth' });

      // Bind deploy buttons
      document.querySelectorAll('.btn-deploy-chk').forEach(btn => {
        btn.addEventListener('click', async () => {
          const chkId = btn.dataset.id;
          const defaultDeployName = `genesis-tuned-${chkId.substring(0, 8)}`;
          const deployName = prompt('Enter model name to register in Ollama:', defaultDeployName);
          if (!deployName) return;

          btn.disabled = true;
          btn.textContent = 'Deploying...';
          try {
            const depRes = await fetch(`/api/training/checkpoints/${chkId}/deploy`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                deployed_name: deployName,
                system_prompt: 'You are an advanced Project Genesis fine-tuned reasoning agent.',
                parameters: { temperature: 0.2, top_p: 0.9 },
              }),
            });
            if (!depRes.ok) throw new Error(await depRes.text());
            const depData = await depRes.json();
            alert(`Checkpoint deployed successfully as '${depData.deployed_name}'!\nModelfile written to: ${depData.modelfile_path}`);
            await viewExperimentDetails(experimentId);
          } catch (err) {
            alert('Deployment failed: ' + err.message);
          } finally {
            btn.disabled = false;
            btn.textContent = '🚀 Deploy to Ollama';
          }
        });
      });
    } catch (e) {
      trainingDetailsContent.innerHTML = `<p class="text-danger">Failed to load experiment details: ${e.message}</p>`;
    }
  }

  // ── Research & Capabilities Subsystem (Phase 6) ──
  let activeResearchSubtab = 'profiles';

  async function loadResearchData() {
    await loadContextBudget();
    await loadResearchHistory(activeResearchSubtab);

    // Populate dataset select
    try {
      const res = await fetch('/api/training/datasets');
      if (res.ok) {
        const data = await res.json();
        const select = document.getElementById('code-export-dataset');
        if (select) {
          select.innerHTML = '<option value="">None (Standalone experiment)</option>';
          (data.datasets || []).forEach(ds => {
            select.innerHTML += `<option value="${ds.id}">${ds.name} (${ds.format})</option>`;
          });
        }
      }
    } catch (e) {
      console.warn('Failed to load datasets for research export:', e);
    }
  }

  async function loadContextBudget() {
    const modelSelect = document.getElementById('budget-model-select');
    const tokensInput = document.getElementById('budget-context-tokens');
    const tokensVal = document.getElementById('budget-tokens-val');
    if (!modelSelect || !tokensInput) return;

    const modelId = modelSelect.value;
    const tokens = parseInt(tokensInput.value, 10);
    if (tokensVal) tokensVal.textContent = tokens;

    try {
      const res = await fetch(`/api/research/context-budget?model_id=${encodeURIComponent(modelId)}&context_tokens=${tokens}`);
      if (!res.ok) return;
      const data = await res.json();

      document.getElementById('metric-kv-cache').textContent = `${data.kv_cache_mb} MB`;
      document.getElementById('metric-weights').textContent = `${data.model_weights_mb} MB`;
      document.getElementById('metric-total-vram').textContent = `${data.total_vram_mb} MB`;
      document.getElementById('metric-headroom').textContent = `${data.headroom_mb} MB`;
      document.getElementById('metric-rec-tokens').textContent = `${data.recommended_max_tokens} tokens`;

      const badge = document.getElementById('budget-safety-badge');
      if (badge) {
        if (data.is_safe) {
          badge.className = 'badge badge-success';
          badge.textContent = 'Safe Budget (Headroom > 1GB)';
        } else {
          badge.className = 'badge badge-danger';
          badge.textContent = 'OOM Risk (Low Headroom)';
        }
      }
    } catch (err) {
      console.error('Failed to calculate context budget:', err);
    }
  }

  const budgetModelSelect = document.getElementById('budget-model-select');
  const budgetContextTokens = document.getElementById('budget-context-tokens');
  if (budgetModelSelect) budgetModelSelect.addEventListener('change', loadContextBudget);
  if (budgetContextTokens) budgetContextTokens.addEventListener('input', loadContextBudget);

  const btnRefreshResearch = document.getElementById('btn-refresh-research');
  if (btnRefreshResearch) btnRefreshResearch.addEventListener('click', loadResearchData);

  // Live Profiler Button
  const btnRunProfile = document.getElementById('btn-run-profile');
  if (btnRunProfile) {
    btnRunProfile.addEventListener('click', async () => {
      const modelSelect = document.getElementById('budget-model-select');
      const promptInput = document.getElementById('profile-prompt-input');
      const resultsDiv = document.getElementById('profile-live-results');

      btnRunProfile.disabled = true;
      btnRunProfile.textContent = '⚡ Profiling GPU Tokens...';

      try {
        const res = await fetch('/api/research/profile', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_id: modelSelect ? modelSelect.value : 'qwen2.5:7b',
            prompt: promptInput ? promptInput.value : 'Explain QuickSort in 3 sentences.',
            max_tokens: 64,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        document.getElementById('live-ttft').textContent = `${data.ttft_ms} ms`;
        document.getElementById('live-tps').textContent = `${data.tokens_per_second} tok/s`;
        document.getElementById('live-itl').textContent = `${data.inter_token_latency_ms} ms`;
        if (resultsDiv) resultsDiv.style.display = 'block';

        if (activeResearchSubtab === 'profiles') await loadResearchHistory('profiles');
      } catch (err) {
        alert('Profiling failed: ' + err.message);
      } finally {
        btnRunProfile.disabled = false;
        btnRunProfile.textContent = '⚡ Profile Inference Speed';
      }
    });
  }

  // Code Synthesizer Form
  const codeSynthForm = document.getElementById('code-synth-form');
  if (codeSynthForm) {
    codeSynthForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const objInput = document.getElementById('code-objective');
      const maxIterSelect = document.getElementById('code-max-iterations');
      const expDsSelect = document.getElementById('code-export-dataset');
      const submitBtn = document.getElementById('btn-submit-code-synth');
      const resultPanel = document.getElementById('code-result-panel');
      const outcomeBadge = document.getElementById('code-outcome-badge');
      const iterBadge = document.getElementById('code-iterations-badge');
      const previewBox = document.getElementById('code-preview-box');
      const tracebackBox = document.getElementById('code-traceback-box');

      submitBtn.disabled = true;
      submitBtn.textContent = '⚙️ Synthesizing & Testing in Sandbox...';

      try {
        const res = await fetch('/api/research/coding/synthesize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            objective: objInput.value,
            max_iterations: parseInt(maxIterSelect.value, 10),
            export_dataset_id: expDsSelect.value || null,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        if (resultPanel) resultPanel.style.display = 'block';
        if (outcomeBadge) {
          outcomeBadge.className = data.status === 'passed' ? 'badge badge-success' : 'badge badge-danger';
          outcomeBadge.textContent = data.status.toUpperCase();
        }
        if (iterBadge) iterBadge.textContent = `Iterations: ${data.iterations}/${data.max_iterations} (${data.duration_ms} ms)`;
        if (previewBox) previewBox.textContent = data.final_code || data.initial_code || 'No code produced';

        if (tracebackBox) {
          if (data.status !== 'passed' && data.test_results && data.test_results.traceback) {
            tracebackBox.style.display = 'block';
            tracebackBox.textContent = data.test_results.traceback;
          } else {
            tracebackBox.style.display = 'none';
          }
        }

        if (activeResearchSubtab === 'code') await loadResearchHistory('code');
      } catch (err) {
        alert('Code synthesis failed: ' + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '⚡ Synthesize & Sandbox Test';
      }
    });
  }

  // Knowledge Synthesizer Form
  const synthSourcesVal = document.getElementById('synth-sources-val');
  const synthMaxSources = document.getElementById('synth-max-sources');
  if (synthMaxSources && synthSourcesVal) {
    synthMaxSources.addEventListener('input', () => {
      synthSourcesVal.textContent = synthMaxSources.value;
    });
  }

  const knowledgeSynthForm = document.getElementById('knowledge-synth-form');
  if (knowledgeSynthForm) {
    knowledgeSynthForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const topicInput = document.getElementById('synth-topic');
      const maxSources = parseInt(synthMaxSources.value, 10);
      const saveMemCheck = document.getElementById('synth-save-memory');
      const submitBtn = document.getElementById('btn-submit-knowledge-synth');
      const resultPanel = document.getElementById('synth-result-panel');
      const groundedBadge = document.getElementById('synth-grounded-badge');
      const previewBox = document.getElementById('synth-markdown-preview');

      submitBtn.disabled = true;
      submitBtn.textContent = '🔬 Synthesizing & Verifying Citations...';

      try {
        const res = await fetch('/api/research/synthesize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            topic: topicInput.value,
            max_sources: maxSources,
            save_to_memory: saveMemCheck ? saveMemCheck.checked : true,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        if (resultPanel) resultPanel.style.display = 'block';
        if (groundedBadge) {
          const pct = Math.round(data.confidence_score * 100);
          groundedBadge.className = pct >= 80 ? 'badge badge-success' : 'badge badge-warning';
          groundedBadge.textContent = `Grounded ${pct}% (${data.sources_count} sources)`;
        }
        if (previewBox) {
          previewBox.innerHTML = `<pre style="white-space: pre-wrap; font-family: inherit;">${data.brief_markdown}</pre>`;
        }

        if (activeResearchSubtab === 'syntheses') await loadResearchHistory('syntheses');
      } catch (err) {
        alert('Knowledge synthesis failed: ' + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '🔬 Synthesize Knowledge Brief';
      }
    });
  }

  // Subnav Tabs for Historical History
  const subtabProfilesBtn = document.getElementById('subtab-profiles-btn');
  const subtabCodeBtn = document.getElementById('subtab-code-btn');
  const subtabSynthesesBtn = document.getElementById('subtab-syntheses-btn');

  function updateSubtabs(activeBtn, tabName) {
    [subtabProfilesBtn, subtabCodeBtn, subtabSynthesesBtn].forEach(b => {
      if (b) b.classList.toggle('active', b === activeBtn);
    });
    activeResearchSubtab = tabName;
    loadResearchHistory(tabName);
  }

  if (subtabProfilesBtn) subtabProfilesBtn.addEventListener('click', () => updateSubtabs(subtabProfilesBtn, 'profiles'));
  if (subtabCodeBtn) subtabCodeBtn.addEventListener('click', () => updateSubtabs(subtabCodeBtn, 'code'));
  if (subtabSynthesesBtn) subtabSynthesesBtn.addEventListener('click', () => updateSubtabs(subtabSynthesesBtn, 'syntheses'));

  async function loadResearchHistory(type) {
    const thead = document.getElementById('research-history-thead');
    const tbody = document.getElementById('research-history-tbody');
    if (!thead || !tbody) return;

    tbody.innerHTML = '<tr><td colspan="5" class="text-center">Loading history...</td></tr>';

    try {
      if (type === 'profiles') {
        thead.innerHTML = '<tr><th>Model</th><th>TTFT</th><th>Throughput</th><th>KV VRAM</th><th>Status</th></tr>';
        const res = await fetch('/api/research/profiles?limit=10');
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        const profiles = data.profiles || [];
        if (!profiles.length) {
          tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No inference profiles recorded yet.</td></tr>';
          return;
        }
        tbody.innerHTML = profiles.map(p => `
          <tr>
            <td><strong>${p.model_id}</strong></td>
            <td>${p.ttft_ms} ms</td>
            <td>${p.tokens_per_second} tok/s</td>
            <td>${p.kv_cache_vram_mb} MB</td>
            <td><span class="badge ${p.status === 'completed' ? 'badge-success' : 'badge-danger'}">${p.status}</span></td>
          </tr>
        `).join('');
      } else if (type === 'code') {
        thead.innerHTML = '<tr><th>Objective</th><th>Iterations</th><th>Duration</th><th>Outcome</th><th>Export</th></tr>';
        const res = await fetch('/api/research/coding/experiments?limit=10');
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        const experiments = data.experiments || [];
        if (!experiments.length) {
          tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No code synthesis experiments recorded yet.</td></tr>';
          return;
        }
        tbody.innerHTML = experiments.map(e => `
          <tr>
            <td title="${e.objective}">${e.objective.substring(0, 40)}...</td>
            <td>${e.iterations}/${e.max_iterations}</td>
            <td>${e.duration_ms} ms</td>
            <td><span class="badge ${e.status === 'passed' ? 'badge-success' : 'badge-danger'}">${e.status}</span></td>
            <td><small>${e.metadata && e.metadata.exported ? 'Dataset' : 'Standalone'}</small></td>
          </tr>
        `).join('');
      } else if (type === 'syntheses') {
        thead.innerHTML = '<tr><th>Topic</th><th>Sources</th><th>Grounding</th><th>Status</th><th>Date</th></tr>';
        const res = await fetch('/api/research/syntheses?limit=10');
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        const syntheses = data.syntheses || [];
        if (!syntheses.length) {
          tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No research syntheses recorded yet.</td></tr>';
          return;
        }
        tbody.innerHTML = syntheses.map(s => `
          <tr>
            <td title="${s.topic}"><strong>${s.topic.substring(0, 35)}...</strong></td>
            <td>${s.sources_count} chunks</td>
            <td><span class="badge badge-info">${Math.round(s.confidence_score * 100)}%</span></td>
            <td><span class="badge ${s.status === 'completed' ? 'badge-success' : 'badge-danger'}">${s.status}</span></td>
            <td><small>${new Date(s.created_at).toLocaleTimeString()}</small></td>
          </tr>
        `).join('');
      }
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="5" class="text-danger">Error loading history: ${err.message}</td></tr>`;
    }
  }

  // ── Initial Bootstrap ──
  loadConversations();
  loadCategories();
  loadHardwareAndModels();
  loadDatasets();
});

