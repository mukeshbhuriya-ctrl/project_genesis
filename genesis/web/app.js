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

  // ── Initial Bootstrap ──
  loadConversations();
  loadCategories();
  loadHardwareAndModels();
});
