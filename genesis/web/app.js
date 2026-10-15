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
      if (target === 'training') { loadTrainingData(); loadReplayBuffer(); loadEWC(); }
      if (target === 'research') { loadResearchData(); loadSymbolicVerifications(); }
      if (target === 'deployment') loadDeploymentData();
      if (target === 'hardware') loadHardwareAndModels();
      if (target === 'backups') loadBackups();
      if (target === 'meta') loadMetaData();
      if (target === 'agents') loadAgentsData();
      if (target === 'graph') loadGraphData();
      if (target === 'fleet') loadFleetData();
      if (target === 'architect') loadArchitectData();
      if (target === 'world') loadWorldData();
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
    await loadSteeringDirectives();

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

  // ── Deployment & Topology Subsystem (Phase 7) ──
  const btnRefreshDeployment = document.getElementById('btn-refresh-deployment');
  const layerPlanForm = document.getElementById('layer-plan-form');
  const planContextTokens = document.getElementById('plan-context-tokens');
  const planContextTokensVal = document.getElementById('plan-context-tokens-val');
  const btnEnqueueTestJob = document.getElementById('btn-enqueue-test-job');

  if (planContextTokens && planContextTokensVal) {
    planContextTokens.addEventListener('input', () => {
      planContextTokensVal.textContent = planContextTokens.value;
    });
  }

  if (btnRefreshDeployment) {
    btnRefreshDeployment.addEventListener('click', () => loadDeploymentData(true));
  }

  if (btnEnqueueTestJob) {
    btnEnqueueTestJob.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/deployment/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job_type: 'custom',
            payload: { action: 'device_topology_benchmark', duration: 0.5 },
            priority: 15,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        await loadWorkJobs();
      } catch (err) {
        alert('Failed to enqueue job: ' + err.message);
      }
    });
  }

  if (layerPlanForm) {
    layerPlanForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const modelSelect = document.getElementById('plan-model-name');
      const selectedOption = modelSelect.options[modelSelect.selectedIndex];
      const layers = parseInt(selectedOption.dataset.layers || '28', 10);
      const params = parseFloat(selectedOption.dataset.params || '7.0');
      const quant = parseInt(document.getElementById('plan-quant-bits').value, 10);
      const context = parseInt(document.getElementById('plan-context-tokens').value, 10);
      const reserve = parseFloat(document.getElementById('plan-safety-reserve').value || '1024');

      const badge = document.getElementById('layer-plan-strategy-badge');
      const resultsDiv = document.getElementById('layer-plan-results');
      const tbody = document.getElementById('layer-plan-tbody');

      badge.textContent = 'Calculating...';
      badge.className = 'badge badge-secondary';

      try {
        const res = await fetch('/api/deployment/layer-plan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_name: modelSelect.value,
            total_layers: layers,
            parameter_size_b: params,
            quant_bits: quant,
            context_tokens: context,
            safety_reserve_mb: reserve,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const plan = await res.json();

        badge.textContent = 'Strategy: ' + plan.strategy.toUpperCase().replace('_', ' ');
        badge.className = plan.strategy === 'single_gpu' ? 'badge badge-success' : (plan.strategy === 'pipeline_parallel' ? 'badge badge-info' : 'badge badge-warning');

        resultsDiv.style.display = 'block';
        tbody.innerHTML = Object.entries(plan.device_layers).map(([devId, count]) => {
          const vram = plan.estimated_vram_usage_mb[devId] || 0;
          return `
            <tr>
              <td><strong>${devId}</strong></td>
              <td>${count} / ${plan.total_layers} layers</td>
              <td>${vram.toFixed(1)} MB</td>
              <td><span class="badge ${count > 0 ? 'badge-success' : 'badge-secondary'}">${count > 0 ? 'Active' : 'Idle'}</span></td>
            </tr>
          `;
        }).join('');
      } catch (err) {
        badge.textContent = 'Error';
        badge.className = 'badge badge-danger';
        alert('Layer planning failed: ' + err.message);
      }
    });
  }

  async function loadDeploymentData(forceRefresh = false) {
    const topoHostname = document.getElementById('topo-hostname');
    const topoTotalVram = document.getElementById('topo-total-vram');
    const topoTotalRam = document.getElementById('topo-total-ram');
    const topoMode = document.getElementById('topo-mode');
    const topoBackendBadge = document.getElementById('deployment-backend-badge');
    const topoDevicesTbody = document.getElementById('topo-devices-tbody');

    try {
      const url = forceRefresh ? '/api/deployment/topology?refresh=true' : '/api/deployment/topology';
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      if (topoHostname) topoHostname.textContent = data.hostname || '--';
      if (topoTotalVram) topoTotalVram.textContent = `${data.total_vram_mb} MB`;
      if (topoTotalRam) topoTotalRam.textContent = `${data.total_ram_gb} GB`;
      if (topoMode) topoMode.textContent = data.is_multi_gpu ? 'Multi-GPU Cluster' : 'Single GPU';
      if (topoBackendBadge) topoBackendBadge.textContent = `Backend: ${data.compute_backend.toUpperCase()}`;

      if (topoDevicesTbody && data.devices) {
        topoDevicesTbody.innerHTML = data.devices.map(d => `
          <tr>
            <td><strong>${d.id}</strong></td>
            <td>${d.name}</td>
            <td><span class="badge ${d.device_type === 'cuda' ? 'badge-success' : 'badge-secondary'}">${d.device_type.toUpperCase()}</span></td>
            <td>${d.vram_total_mb.toFixed(0)} MB total (${d.vram_free_mb.toFixed(0)} MB free)</td>
            <td>${d.is_primary ? '<span class="badge badge-info">Primary</span>' : '<span class="badge badge-secondary">Secondary</span>'}</td>
          </tr>
        `).join('');
      }

      await loadWorkJobs();
    } catch (err) {
      console.error('Error loading deployment topology:', err);
    }
  }

  async function loadWorkJobs() {
    const tbody = document.getElementById('jobs-table-tbody');
    if (!tbody) return;

    try {
      const res = await fetch('/api/deployment/jobs?limit=25');
      if (!res.ok) throw new Error(await res.text());
      const jobs = await res.json();

      if (!jobs || !jobs.length) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">No background jobs in queue.</td></tr>';
        return;
      }

      tbody.innerHTML = jobs.map(j => {
        const statusClass = j.status === 'completed' ? 'badge-success' : (j.status === 'running' || j.status === 'leased' ? 'badge-info' : (j.status === 'failed' ? 'badge-danger' : 'badge-secondary'));
        const canCancel = j.status === 'queued' || j.status === 'leased' || j.status === 'running';
        const progressPct = Math.round((j.progress || 0) * 100);

        return `
          <tr>
            <td><code>${j.id.substring(0, 8)}...</code></td>
            <td><strong>${j.job_type}</strong></td>
            <td><span class="badge ${statusClass}">${j.status.toUpperCase()}</span></td>
            <td>${j.priority}</td>
            <td>
              <div style="background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden; height: 14px; width: 80px;">
                <div style="background: var(--accent-cyan); width: ${progressPct}%; height: 100%;"></div>
              </div>
              <small>${progressPct}%</small>
            </td>
            <td><small>${j.worker_id ? j.worker_id.substring(0, 16) : '--'}</small></td>
            <td><small>${new Date(j.created_at).toLocaleTimeString()}</small></td>
            <td>
              ${canCancel ? `<button class="btn btn-xs btn-danger" onclick="cancelWorkJob('${j.id}')">Cancel</button>` : '<span class="text-muted">--</span>'}
            </td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="8" class="text-danger">Error loading work jobs: ${err.message}</td></tr>`;
    }
  }

  window.cancelWorkJob = async function(jobId) {
    if (!confirm('Cancel work job ' + jobId + '?')) return;
    try {
      const res = await fetch(`/api/deployment/jobs/${jobId}/cancel`, { method: 'POST' });
      if (!res.ok) throw new Error(await res.text());
      await loadWorkJobs();
    } catch (err) {
      alert('Failed to cancel job: ' + err.message);
    }
  };

  // ── Meta-Improvement & Multi-Model Dispatch Subsystem ──
  const metaCycleModal = document.getElementById('meta-cycle-modal');
  const btnOpenMetaCycleModal = document.getElementById('btn-open-meta-cycle-modal');
  const btnCloseMetaModal = document.getElementById('btn-close-meta-modal');
  const btnCancelMetaCycle = document.getElementById('btn-cancel-meta-cycle');
  const launchMetaCycleForm = document.getElementById('launch-meta-cycle-form');
  const testRouterForm = document.getElementById('test-router-form');
  const btnTriggerExpansion = document.getElementById('btn-trigger-expansion');

  if (btnOpenMetaCycleModal && metaCycleModal) {
    btnOpenMetaCycleModal.addEventListener('click', () => {
      const nameEl = document.getElementById('meta-cycle-name');
      if (nameEl) nameEl.value = 'auto-cycle-' + Date.now().toString().slice(-6);
      metaCycleModal.style.display = 'flex';
    });
  }

  if (btnCloseMetaModal && metaCycleModal) {
    btnCloseMetaModal.addEventListener('click', () => {
      metaCycleModal.style.display = 'none';
    });
  }

  if (btnCancelMetaCycle && metaCycleModal) {
    btnCancelMetaCycle.addEventListener('click', () => {
      metaCycleModal.style.display = 'none';
    });
  }

  async function loadMetaData() {
    await Promise.all([
      loadMetaCycles(),
      loadModelRoutes()
    ]);
  }

  async function loadMetaCycles() {
    const tbody = document.getElementById('meta-cycles-tbody');
    if (!tbody) return;

    try {
      const res = await fetch('/api/meta/cycles');
      if (!res.ok) throw new Error(await res.text());
      const cycles = await res.json();

      if (!cycles || !cycles.length) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No evolution cycles recorded yet.</td></tr>';
        return;
      }

      tbody.innerHTML = cycles.map(c => {
        let statusBadge = 'badge-secondary';
        if (c.status === 'completed') statusBadge = 'badge-success';
        else if (c.status === 'failed') statusBadge = 'badge-danger';
        else if (c.status === 'running') statusBadge = 'badge-info';

        let verdictBadge = 'badge-secondary';
        if (c.decision_verdict === 'promote') verdictBadge = 'badge-success';
        else if (c.decision_verdict === 'rollback') verdictBadge = 'badge-warning';

        const baseScore = c.baseline_score !== null && c.baseline_score !== undefined ? (c.baseline_score * 100).toFixed(1) + '%' : '--';
        const candScore = c.candidate_score !== null && c.candidate_score !== undefined ? (c.candidate_score * 100).toFixed(1) + '%' : '--';

        return `
          <tr>
            <td><strong>${c.cycle_name || c.id.substring(0, 8)}</strong></td>
            <td><code>${c.benchmark_suite_id || '--'}</code></td>
            <td><span class="badge ${statusBadge}">${c.status.toUpperCase()}</span></td>
            <td><small>Base: ${baseScore} &rarr; Cand: ${candScore}</small></td>
            <td><span class="badge ${verdictBadge}">${(c.decision_verdict || '--').toUpperCase()}</span></td>
            <td><small>${new Date(c.created_at).toLocaleString()}</small></td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="6" class="text-danger">Failed to load cycles: ${err.message}</td></tr>`;
    }
  }

  async function loadModelRoutes() {
    const tbody = document.getElementById('routes-tbody');
    if (!tbody) return;

    try {
      const res = await fetch('/api/meta/routes');
      if (!res.ok) throw new Error(await res.text());
      const routes = await res.json();

      if (!routes || !routes.length) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No routes configured.</td></tr>';
        return;
      }

      tbody.innerHTML = routes.map(r => `
        <tr>
          <td><span class="badge badge-info">${r.task_category}</span></td>
          <td><strong>${r.primary_model}</strong></td>
          <td>${r.fallback_model ? `<code>${r.fallback_model}</code>` : '<span class="text-muted">None</span>'}</td>
          <td><small>${r.min_vram_gb} GB</small></td>
        </tr>
      `).join('');
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="4" class="text-danger">Failed to load routes: ${err.message}</td></tr>`;
    }
  }

  if (testRouterForm) {
    testRouterForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const prompt = document.getElementById('route-test-prompt').value.trim();
      const catOverride = document.getElementById('route-override-category').value.trim();
      const resultBox = document.getElementById('route-decision-result');
      const modelNameEl = document.getElementById('routed-model-name');
      const catBadgeEl = document.getElementById('routed-category-badge');
      const reasonEl = document.getElementById('routed-reason-text');

      if (!prompt) return;

      try {
        const payload = { prompt };
        if (catOverride) payload.task_category = catOverride;

        const res = await fetch('/api/meta/route', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        resultBox.style.display = 'block';
        modelNameEl.textContent = data.selected_model + (data.is_fallback ? ' (Fallback)' : '');
        catBadgeEl.textContent = data.task_category.toUpperCase();
        reasonEl.textContent = `Routing rationale: ${data.routing_reason} (VRAM Headroom: ${data.free_vram_gb.toFixed(2)} GB, Fallback: ${data.is_fallback ? 'YES' : 'NO'})`;
      } catch (err) {
        alert('Route query failed: ' + err.message);
      }
    });
  }

  if (btnTriggerExpansion) {
    btnTriggerExpansion.addEventListener('click', async () => {
      const suiteId = document.getElementById('expand-target-bm').value;
      const domain = document.getElementById('expand-domain').value;
      const count = parseInt(document.getElementById('expand-count').value, 10) || 3;
      const resultBox = document.getElementById('expansion-result-box');
      const summaryText = document.getElementById('expansion-summary-text');
      const casesTbody = document.getElementById('expansion-cases-tbody');

      btnTriggerExpansion.disabled = true;
      btnTriggerExpansion.textContent = '⚡ Expanding...';

      try {
        const res = await fetch('/api/meta/benchmark/expand', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            target_suite_id: suiteId,
            domain: domain,
            num_cases: count
          })
        });

        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        resultBox.style.display = 'block';
        summaryText.textContent = `Synthesized ${data.synthesized_cases_count} cases for suite "${data.target_suite_id}" from ${data.mined_failures_count} failure edge-case traces.`;

        if (data.synthesized_cases && data.synthesized_cases.length) {
          casesTbody.innerHTML = data.synthesized_cases.map(c => `
            <tr>
              <td><code>${c.name}</code></td>
              <td><small>${(c.prompt || '').substring(0, 60)}...</small></td>
              <td><code>${(c.expected_output || '').substring(0, 30)}...</code></td>
              <td><span class="badge badge-info">${c.grading_type}</span></td>
            </tr>
          `).join('');
        } else {
          casesTbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No cases rendered.</td></tr>';
        }
      } catch (err) {
        alert('Benchmark expansion failed: ' + err.message);
      } finally {
        btnTriggerExpansion.disabled = false;
        btnTriggerExpansion.textContent = '⚡ Expand Benchmark Suite';
      }
    });
  }

  if (launchMetaCycleForm) {
    launchMetaCycleForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('btn-submit-meta-cycle');
      const cycleName = document.getElementById('meta-cycle-name').value.trim();
      const suiteId = document.getElementById('meta-benchmark-select').value;
      const baseModel = document.getElementById('meta-base-model-select').value;
      const steps = parseInt(document.getElementById('meta-training-steps').value, 10) || 20;
      const regThreshold = parseFloat(document.getElementById('meta-regression-threshold').value) || 0.05;

      submitBtn.disabled = true;
      submitBtn.textContent = '🚀 Launching Cycle...';

      try {
        const res = await fetch('/api/meta/cycles/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            cycle_name: cycleName,
            benchmark_suite_id: suiteId,
            target_base_model: baseModel,
            training_steps: steps,
            regression_threshold: regThreshold
          })
        });

        if (!res.ok) throw new Error(await res.text());
        metaCycleModal.style.display = 'none';
        await loadMetaCycles();
        alert('Autonomous evolution cycle launched successfully!');
      } catch (err) {
        alert('Cycle execution failed: ' + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '🚀 Launch Closed-Loop Cycle';
      }
    });
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

  // ── Multi-Agent & Campaign Subsystem ──
  const agentsRosterGrid = document.getElementById('agents-roster-grid');
  const debatesTbody = document.getElementById('debates-tbody');
  const campaignsTbody = document.getElementById('campaigns-tbody');
  const debateModal = document.getElementById('debate-modal');
  const campaignModal = document.getElementById('campaign-modal');
  const btnOpenDebateModal = document.getElementById('btn-open-debate-modal');
  const btnCloseDebateModal = document.getElementById('btn-close-debate-modal');
  const btnCancelDebate = document.getElementById('btn-cancel-debate');
  const launchDebateForm = document.getElementById('launch-debate-form');
  const btnOpenCampaignModal = document.getElementById('btn-open-campaign-modal');
  const btnCloseCampaignModal = document.getElementById('btn-close-campaign-modal');
  const btnCancelCampaign = document.getElementById('btn-cancel-campaign');
  const launchCampaignForm = document.getElementById('launch-campaign-form');
  const btnRefreshDebates = document.getElementById('btn-refresh-debates');
  const btnRefreshCampaigns = document.getElementById('btn-refresh-campaigns');
  const debateDetailPanel = document.getElementById('debate-detail-panel');
  const btnCloseDebateDetail = document.getElementById('btn-close-debate-detail');
  const campaignDetailPanel = document.getElementById('campaign-detail-panel');
  const btnCloseCampaignDetail = document.getElementById('btn-close-campaign-detail');

  const ROLE_ICONS = {
    researcher: '🔬',
    architect: '📐',
    implementer: '💻',
    reviewer: '🛡️',
    synthesizer: '⚖️'
  };

  async function loadAgentsData() {
    await Promise.all([
      loadAgentRoles(),
      loadAgentTeams(),
      loadDebateSessions(),
      loadResearchCampaigns()
    ]);
  }

  async function loadAgentRoles() {
    if (!agentsRosterGrid) return;
    try {
      const res = await fetch('/api/agents/roles');
      if (!res.ok) throw new Error(await res.text());
      const roles = await res.json();
      if (!roles || roles.length === 0) {
        agentsRosterGrid.innerHTML = '<div class="text-muted">No roles configured.</div>';
        return;
      }
      agentsRosterGrid.innerHTML = roles.map(r => {
        const icon = ROLE_ICONS[r.role] || '🤖';
        return `
          <div class="roster-role-card">
            <div class="roster-role-header">
              <span class="roster-role-title">${icon} ${r.role.toUpperCase()}</span>
              <span class="badge badge-info" style="font-size:0.7rem;">Active</span>
            </div>
            <div class="roster-role-desc">${escapeHtml(r.description || '')}</div>
            <div style="font-size:0.75rem; color:var(--text-dim); margin-top:4px;">
              <code>${escapeHtml((r.prompt_preview || '').substring(0, 85))}...</code>
            </div>
          </div>
        `;
      }).join('');
    } catch (err) {
      agentsRosterGrid.innerHTML = `<div class="text-danger">Failed to load roles: ${escapeHtml(err.message)}</div>`;
    }
  }

  async function loadAgentTeams() {
    try {
      const res = await fetch('/api/agents/teams');
      if (!res.ok) throw new Error(await res.text());
      const teams = await res.json();
      const debateSelect = document.getElementById('debate-team-select');
      const campaignSelect = document.getElementById('campaign-team-select');
      if (teams && teams.length > 0) {
        const optionsHtml = teams.map(t => `<option value="${t.id}">${escapeHtml(t.name)} (${(t.members || []).length} agents)</option>`).join('');
        if (debateSelect) debateSelect.innerHTML = optionsHtml;
        if (campaignSelect) campaignSelect.innerHTML = optionsHtml;
        const activeBadge = document.getElementById('active-team-badge');
        if (activeBadge) activeBadge.textContent = teams[0].name;
      }
    } catch (err) {
      console.warn('Could not load teams:', err);
    }
  }

  async function loadDebateSessions() {
    if (!debatesTbody) return;
    try {
      const res = await fetch('/api/agents/debates');
      if (!res.ok) throw new Error(await res.text());
      const debates = await res.json();
      if (!debates || debates.length === 0) {
        debatesTbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No debate sessions yet. Click "Consensus Debate" to convene one!</td></tr>';
        return;
      }
      debatesTbody.innerHTML = debates.map(d => {
        const verdictSnippet = d.consensus_verdict ? (d.consensus_verdict.substring(0, 50) + '...') : '<span class="text-muted">In Progress</span>';
        const statusBadge = d.status === 'concluded' ? '<span class="badge badge-success">Concluded</span>' : '<span class="badge badge-warning">Active</span>';
        return `
          <tr>
            <td><strong>${escapeHtml(d.topic || '')}</strong></td>
            <td><code>${escapeHtml(d.team_id || 'default')}</code></td>
            <td>${d.rounds || 1}</td>
            <td>${statusBadge}</td>
            <td><small>${verdictSnippet}</small></td>
            <td>
              <button class="btn btn-sm btn-secondary" onclick="window.viewDebateDetails('${d.id}')">View Turns</button>
            </td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      debatesTbody.innerHTML = `<tr><td colspan="6" class="text-danger">Failed to load debates: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  window.viewDebateDetails = async function(debateId) {
    if (!debateDetailPanel) return;
    try {
      const res = await fetch(`/api/agents/debates/${debateId}`);
      if (!res.ok) throw new Error(await res.text());
      const debate = await res.json();
      debateDetailPanel.style.display = 'block';

      const titleEl = document.getElementById('debate-detail-title');
      if (titleEl) titleEl.textContent = `Debate: "${debate.topic}"`;

      const verdictBox = document.getElementById('debate-verdict-box');
      if (verdictBox) {
        let dissentingHtml = '';
        if (debate.dissenting_views && debate.dissenting_views.length > 0) {
          dissentingHtml = `<div style="margin-top:8px; font-size:0.8rem; color:#f59e0b;"><strong>Dissenting Opinions:</strong> ${debate.dissenting_views.map(escapeHtml).join('; ')}</div>`;
        }
        verdictBox.innerHTML = `
          <div><strong>Consensus Verdict:</strong> ${escapeHtml(debate.consensus_verdict || 'Awaiting synthesis.')}</div>
          ${dissentingHtml}
        `;
      }

      const turnsStream = document.getElementById('debate-turns-stream');
      if (turnsStream) {
        const turns = debate.turns || [];
        if (turns.length === 0) {
          turnsStream.innerHTML = '<div class="text-muted">No turns recorded yet.</div>';
        } else {
          turnsStream.innerHTML = turns.map(t => {
            const icon = ROLE_ICONS[t.speaker_role] || '🤖';
            const roleClass = `role-${t.speaker_role}`;
            return `
              <div class="debate-turn-card ${roleClass}">
                <div class="debate-turn-header">
                  <span><strong>${icon} ${t.speaker_role.toUpperCase()}</strong> (${escapeHtml(t.agent_name || '')})</span>
                  <span class="badge badge-info" style="font-size:0.7rem;">Round ${t.round_num}</span>
                </div>
                <div class="debate-turn-content">${escapeHtml(t.content || '')}</div>
              </div>
            `;
          }).join('');
        }
      }
      debateDetailPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (err) {
      alert('Failed to load debate turns: ' + err.message);
    }
  };

  async function loadResearchCampaigns() {
    if (!campaignsTbody) return;
    try {
      const res = await fetch('/api/agents/campaigns');
      if (!res.ok) throw new Error(await res.text());
      const campaigns = await res.json();
      if (!campaigns || campaigns.length === 0) {
        campaignsTbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No research campaigns active. Click "New Campaign" to launch!</td></tr>';
        return;
      }
      campaignsTbody.innerHTML = campaigns.map(c => {
        const milestones = c.milestones || [];
        const completedCount = milestones.filter(m => m.status === 'completed').length;
        const totalCount = milestones.length || 4;
        const progressPct = Math.round((completedCount / totalCount) * 100);
        const statusBadge = c.status === 'completed'
          ? '<span class="badge badge-success">Completed</span>'
          : (c.status === 'active' ? '<span class="badge badge-info">Active</span>' : '<span class="badge badge-warning">Draft</span>');
        const advanceBtn = (c.status !== 'completed' && c.status !== 'failed')
          ? `<button class="btn btn-sm btn-primary" onclick="window.advanceCampaign('${c.id}')">⚡ Advance Step</button>`
          : '';
        return `
          <tr>
            <td><strong>${escapeHtml(c.title || '')}</strong></td>
            <td>${statusBadge}</td>
            <td><code>${c.current_milestone_index + 1}/${totalCount}: ${c.milestones && c.milestones[c.current_milestone_index] ? escapeHtml(c.milestones[c.current_milestone_index].name) : 'Done'}</code></td>
            <td>
              <div style="display:flex; align-items:center; gap:8px;">
                <div style="flex:1; background:rgba(255,255,255,0.1); height:6px; border-radius:3px; overflow:hidden;">
                  <div style="width:${progressPct}%; background:#38bdf8; height:100%;"></div>
                </div>
                <small>${progressPct}%</small>
              </div>
            </td>
            <td>
              <button class="btn btn-sm btn-secondary" onclick="window.viewCampaignDetails('${c.id}')" style="margin-right:4px;">View</button>
              ${advanceBtn}
            </td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      campaignsTbody.innerHTML = `<tr><td colspan="5" class="text-danger">Failed to load campaigns: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  window.viewCampaignDetails = async function(campaignId) {
    if (!campaignDetailPanel) return;
    try {
      const res = await fetch(`/api/agents/campaigns/${campaignId}`);
      if (!res.ok) throw new Error(await res.text());
      const campaign = await res.json();
      campaignDetailPanel.style.display = 'block';

      const titleEl = document.getElementById('campaign-detail-title');
      if (titleEl) titleEl.textContent = `Campaign: "${campaign.title}"`;

      const stepper = document.getElementById('campaign-milestones-stepper');
      if (stepper) {
        const milestones = campaign.milestones || [];
        stepper.innerHTML = milestones.map((m, idx) => {
          const isActive = idx === campaign.current_milestone_index && campaign.status !== 'completed';
          const isDone = m.status === 'completed';
          const cls = isDone ? 'completed' : (isActive ? 'active' : '');
          const statusText = isDone ? '✓ Completed' : (isActive ? '● Active' : '○ Pending');
          return `
            <div class="milestone-step ${cls}">
              <div class="milestone-step-name">${idx + 1}. ${escapeHtml(m.name)}</div>
              <div class="milestone-step-status">${statusText}</div>
            </div>
          `;
        }).join('');
      }

      const artifactsBox = document.getElementById('campaign-artifacts-box');
      if (artifactsBox) {
        artifactsBox.textContent = `Goal: ${campaign.goal || ''}\n\n` +
          `Artifacts:\n${JSON.stringify(campaign.artifacts || {}, null, 2)}`;
      }

      const actionBar = document.getElementById('campaign-action-bar');
      if (actionBar) {
        if (campaign.status !== 'completed' && campaign.status !== 'failed') {
          actionBar.innerHTML = `<button class="btn btn-primary" onclick="window.advanceCampaign('${campaign.id}')">⚡ Execute Current Milestone</button>`;
        } else {
          actionBar.innerHTML = `<span class="badge badge-success">Research Campaign Concluded</span>`;
        }
      }
      campaignDetailPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (err) {
      alert('Failed to load campaign: ' + err.message);
    }
  };

  window.advanceCampaign = async function(campaignId) {
    try {
      const res = await fetch(`/api/agents/campaigns/${campaignId}/advance`, {
        method: 'POST'
      });
      if (!res.ok) throw new Error(await res.text());
      await loadResearchCampaigns();
      await window.viewCampaignDetails(campaignId);
    } catch (err) {
      alert('Failed to advance campaign: ' + err.message);
    }
  };

  // Wire Debates UI
  if (btnOpenDebateModal && debateModal) {
    btnOpenDebateModal.addEventListener('click', () => {
      debateModal.style.display = 'flex';
    });
  }
  if (btnCloseDebateModal && debateModal) {
    btnCloseDebateModal.addEventListener('click', () => {
      debateModal.style.display = 'none';
    });
  }
  if (btnCancelDebate && debateModal) {
    btnCancelDebate.addEventListener('click', () => {
      debateModal.style.display = 'none';
    });
  }
  if (btnRefreshDebates) {
    btnRefreshDebates.addEventListener('click', loadDebateSessions);
  }
  if (btnCloseDebateDetail && debateDetailPanel) {
    btnCloseDebateDetail.addEventListener('click', () => {
      debateDetailPanel.style.display = 'none';
    });
  }

  if (launchDebateForm) {
    launchDebateForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('btn-submit-debate');
      const topic = document.getElementById('debate-topic-input').value.trim();
      const teamId = document.getElementById('debate-team-select').value;
      const rounds = parseInt(document.getElementById('debate-rounds-input').value, 10) || 2;
      const context = document.getElementById('debate-context-input').value.trim();

      submitBtn.disabled = true;
      submitBtn.textContent = '⚖️ Convening Agent Debate...';

      try {
        const res = await fetch('/api/agents/debate/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            topic,
            team_id: teamId,
            rounds,
            context
          })
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        debateModal.style.display = 'none';
        launchDebateForm.reset();
        await loadDebateSessions();
        if (data.id) {
          await window.viewDebateDetails(data.id);
        }
      } catch (err) {
        alert('Debate execution failed: ' + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '⚖️ Convene Agent Debate';
      }
    });
  }

  // Wire Campaigns UI
  if (btnOpenCampaignModal && campaignModal) {
    btnOpenCampaignModal.addEventListener('click', () => {
      campaignModal.style.display = 'flex';
    });
  }
  if (btnCloseCampaignModal && campaignModal) {
    btnCloseCampaignModal.addEventListener('click', () => {
      campaignModal.style.display = 'none';
    });
  }
  if (btnCancelCampaign && campaignModal) {
    btnCancelCampaign.addEventListener('click', () => {
      campaignModal.style.display = 'none';
    });
  }
  if (btnRefreshCampaigns) {
    btnRefreshCampaigns.addEventListener('click', loadResearchCampaigns);
  }
  if (btnCloseCampaignDetail && campaignDetailPanel) {
    btnCloseCampaignDetail.addEventListener('click', () => {
      campaignDetailPanel.style.display = 'none';
    });
  }

  if (launchCampaignForm) {
    launchCampaignForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('btn-submit-campaign');
      const title = document.getElementById('campaign-title-input').value.trim();
      const goal = document.getElementById('campaign-goal-input').value.trim();
      const initialHypothesis = document.getElementById('campaign-hypothesis-input').value.trim();
      const teamId = document.getElementById('campaign-team-select').value;

      submitBtn.disabled = true;
      submitBtn.textContent = '🚀 Launching Campaign...';

      try {
        const res = await fetch('/api/agents/campaigns', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title,
            goal,
            team_id: teamId,
            initial_hypothesis: initialHypothesis
          })
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        campaignModal.style.display = 'none';
        launchCampaignForm.reset();
        await loadResearchCampaigns();
        if (data.id) {
          await window.viewCampaignDetails(data.id);
        }
      } catch (err) {
        alert('Campaign launch failed: ' + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '🚀 Launch Research Campaign';
      }
    });
  }

  // ── Phase 10: Knowledge Graph & Autonomous Research Controller ──
  const btnRefreshGraph = document.getElementById('btn-refresh-graph');
  const btnSearchGraph = document.getElementById('btn-search-graph');
  const graphSearchInput = document.getElementById('graph-search-input');
  const graphTypeFilter = document.getElementById('graph-type-filter');
  const graphNodesContainer = document.getElementById('graph-nodes-container');
  const graphNodesCount = document.getElementById('graph-nodes-count');
  const graphEdgesCount = document.getElementById('graph-edges-count');
  const btnFindPath = document.getElementById('btn-find-path');
  const graphPathSource = document.getElementById('graph-path-source');
  const graphPathTarget = document.getElementById('graph-path-target');
  const graphPathResult = document.getElementById('graph-path-result');
  const btnExtractSubgraph = document.getElementById('btn-extract-subgraph');
  const graphSubgraphCenter = document.getElementById('graph-subgraph-center');
  const btnSyncCampaignsGraph = document.getElementById('btn-sync-campaigns-graph');

  const btnRefreshPapers = document.getElementById('btn-refresh-papers');
  const papersTableBody = document.getElementById('papers-table-body');
  const paperDetailCard = document.getElementById('paper-detail-card');
  const paperDetailTitle = document.getElementById('paper-detail-title');
  const paperDetailAbstract = document.getElementById('paper-detail-abstract');
  const paperDetailContent = document.getElementById('paper-detail-content');
  const btnClosePaperDetail = document.getElementById('btn-close-paper-detail');

  const btnRefreshInspections = document.getElementById('btn-refresh-inspections');
  const inspectionsTableBody = document.getElementById('inspections-table-body');
  const inspectionDetailCard = document.getElementById('inspection-detail-card');
  const inspectionDetailName = document.getElementById('inspection-detail-name');
  const inspectionDetailBadge = document.getElementById('inspection-detail-badge');
  const inspectionDetailReport = document.getElementById('inspection-detail-report');

  const btnOpenDaemonModal = document.getElementById('btn-open-daemon-modal');
  const daemonModal = document.getElementById('daemon-modal');
  const btnCloseDaemonModal = document.getElementById('btn-close-daemon-modal');
  const btnCancelDaemon = document.getElementById('btn-cancel-daemon');
  const launchDaemonForm = document.getElementById('launch-daemon-form');

  const btnOpenVisionModal = document.getElementById('btn-open-vision-modal');
  const visionModal = document.getElementById('vision-modal');
  const btnCloseVisionModal = document.getElementById('btn-close-vision-modal');
  const btnCancelVision = document.getElementById('btn-cancel-vision');
  const inspectVisionForm = document.getElementById('inspect-vision-form');

  async function loadGraphData() {
    await Promise.all([loadGraphStats(), loadKnowledgeNodes(), loadPapers(), loadInspections()]);
  }

  async function loadGraphStats() {
    try {
      const res = await fetch('/api/graph/export');
      if (!res.ok) return;
      const data = await res.json();
      if (graphNodesCount && data.stats) {
        graphNodesCount.textContent = `${data.stats.total_nodes} Nodes`;
      }
      if (graphEdgesCount && data.stats) {
        graphEdgesCount.textContent = `${data.stats.total_edges} Edges`;
      }
    } catch (e) {
      console.warn('Failed to load graph stats:', e);
    }
  }

  async function loadKnowledgeNodes() {
    if (!graphNodesContainer) return;
    try {
      const type = graphTypeFilter ? graphTypeFilter.value : '';
      const url = type ? `/api/graph/nodes?node_type=${encodeURIComponent(type)}` : '/api/graph/nodes';
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      const nodes = await res.json();

      const search = graphSearchInput ? graphSearchInput.value.trim().toLowerCase() : '';
      const filtered = search
        ? nodes.filter(n => (n.label || '').toLowerCase().includes(search) || (n.id || '').toLowerCase().includes(search))
        : nodes;

      if (!filtered.length) {
        graphNodesContainer.innerHTML = '<div class="text-center text-muted" style="padding:40px;">No matching knowledge nodes found.</div>';
        return;
      }

      graphNodesContainer.innerHTML = `
        <div class="graph-nodes-grid">
          ${filtered.map(node => {
            const propsStr = node.properties && Object.keys(node.properties).length
              ? Object.entries(node.properties).map(([k, v]) => `${k}: ${v}`).join(' | ')
              : 'No custom properties';
            return `
              <div class="graph-node-card node-type-${escapeHtml(node.node_type || 'concept')}">
                <div class="node-header">
                  <div class="node-label">${escapeHtml(node.label || node.id)}</div>
                  <span class="badge badge-secondary">${escapeHtml(node.node_type || 'node')}</span>
                </div>
                <div class="node-id">${escapeHtml(node.id)}</div>
                <div class="node-props">${escapeHtml(propsStr)}</div>
                <div class="node-actions">
                  <button class="btn btn-xs btn-secondary" onclick="window.setPathNode('${escapeHtml(node.id)}', 'source')">Src</button>
                  <button class="btn btn-xs btn-secondary" onclick="window.setPathNode('${escapeHtml(node.id)}', 'target')">Dst</button>
                  <button class="btn btn-xs btn-secondary" onclick="window.setPathNode('${escapeHtml(node.id)}', 'center')">Center</button>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      `;
    } catch (err) {
      graphNodesContainer.innerHTML = `<div class="text-danger" style="padding:20px;">Error loading nodes: ${escapeHtml(err.message)}</div>`;
    }
  }

  window.setPathNode = function(id, role) {
    if (role === 'source' && graphPathSource) graphPathSource.value = id;
    if (role === 'target' && graphPathTarget) graphPathTarget.value = id;
    if (role === 'center' && graphSubgraphCenter) graphSubgraphCenter.value = id;
  };

  async function loadPapers() {
    if (!papersTableBody) return;
    try {
      const res = await fetch('/api/research/daemon/papers');
      if (!res.ok) throw new Error(await res.text());
      const papers = await res.json();
      if (!papers.length) {
        papersTableBody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No publications synthesized yet. Launch a research cycle to begin.</td></tr>';
        return;
      }

      papersTableBody.innerHTML = papers.map(p => `
        <tr>
          <td><strong>${escapeHtml(p.title)}</strong><br><small style="color:var(--text-muted);">${escapeHtml((p.abstract || '').slice(0, 80))}...</small></td>
          <td><span class="badge badge-success">${escapeHtml(p.status || 'published')}</span></td>
          <td><small>${escapeHtml((p.created_at || '').slice(0, 16).replace('T', ' '))}</small></td>
          <td><button class="btn btn-xs btn-primary" onclick="window.viewPaperDetails('${escapeHtml(p.id)}')">📖 Read IMRaD</button></td>
        </tr>
      `).join('');
    } catch (err) {
      papersTableBody.innerHTML = `<tr><td colspan="4" class="text-danger">Failed to load papers: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  window.viewPaperDetails = async function(id) {
    try {
      const res = await fetch(`/api/research/daemon/papers/${id}`);
      if (!res.ok) throw new Error(await res.text());
      const paper = await res.json();
      if (paperDetailTitle) paperDetailTitle.textContent = paper.title || id;
      if (paperDetailAbstract) paperDetailAbstract.textContent = paper.abstract || 'No abstract.';
      if (paperDetailContent) paperDetailContent.textContent = paper.content_markdown || 'No content.';
      if (paperDetailCard) paperDetailCard.style.display = 'block';
      paperDetailCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (e) {
      alert('Error fetching paper details: ' + e.message);
    }
  };

  async function loadInspections() {
    if (!inspectionsTableBody) return;
    try {
      const res = await fetch('/api/multimodal/inspections');
      if (!res.ok) throw new Error(await res.text());
      const inspections = await res.json();
      if (!inspections.length) {
        inspectionsTableBody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No visual artifacts inspected yet.</td></tr>';
        return;
      }

      inspectionsTableBody.innerHTML = inspections.map(i => {
        const passedBadge = i.passed
          ? '<span class="badge badge-success">PASSED</span>'
          : '<span class="badge badge-danger">FAILED</span>';
        const detected = Array.isArray(i.elements_detected) ? i.elements_detected.join(', ') : 'None';
        return `
          <tr>
            <td><strong>${escapeHtml(i.artifact_name)}</strong></td>
            <td><span class="badge badge-secondary">${escapeHtml(i.artifact_type)}</span></td>
            <td><code>${escapeHtml(i.format || 'unknown')}</code></td>
            <td>${passedBadge}</td>
            <td><small>${escapeHtml(detected)}</small></td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      inspectionsTableBody.innerHTML = `<tr><td colspan="5" class="text-danger">Failed to load inspections: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  // Wire Graph Event Listeners
  if (btnRefreshGraph) btnRefreshGraph.addEventListener('click', loadGraphData);
  if (btnSearchGraph) btnSearchGraph.addEventListener('click', loadKnowledgeNodes);
  if (graphTypeFilter) graphTypeFilter.addEventListener('change', loadKnowledgeNodes);
  if (btnRefreshPapers) btnRefreshPapers.addEventListener('click', loadPapers);
  if (btnClosePaperDetail && paperDetailCard) {
    btnClosePaperDetail.addEventListener('click', () => {
      paperDetailCard.style.display = 'none';
    });
  }
  if (btnRefreshInspections) btnRefreshInspections.addEventListener('click', loadInspections);

  if (btnFindPath) {
    btnFindPath.addEventListener('click', async () => {
      const s = graphPathSource ? graphPathSource.value.trim() : '';
      const t = graphPathTarget ? graphPathTarget.value.trim() : '';
      if (!s || !t) {
        alert('Please specify both source and target node IDs.');
        return;
      }
      btnFindPath.disabled = true;
      btnFindPath.textContent = 'Searching path...';
      try {
        const res = await fetch(`/api/graph/path?source_id=${encodeURIComponent(s)}&target_id=${encodeURIComponent(t)}`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        if (graphPathResult) {
          if (!data.path || !data.path.length) {
            graphPathResult.innerHTML = '<span class="text-muted">No connecting path found.</span>';
          } else {
            graphPathResult.innerHTML = `
              <strong>Path (${data.length} steps):</strong><br>
              ${data.path.map(step => `<span class="path-pill">${escapeHtml(step)}</span>`).join(' &rarr; ')}
            `;
          }
        }
      } catch (err) {
        if (graphPathResult) graphPathResult.innerHTML = `<span class="text-danger">Path search failed: ${escapeHtml(err.message)}</span>`;
      } finally {
        btnFindPath.disabled = false;
        btnFindPath.textContent = '🧭 Find Shortest Path';
      }
    });
  }

  if (btnExtractSubgraph) {
    btnExtractSubgraph.addEventListener('click', async () => {
      const center = graphSubgraphCenter ? graphSubgraphCenter.value.trim() : '';
      if (!center) {
        alert('Please enter a center node ID for the subgraph.');
        return;
      }
      btnExtractSubgraph.disabled = true;
      try {
        const res = await fetch(`/api/graph/subgraph?centers=${encodeURIComponent(center)}&depth=1`);
        if (!res.ok) throw new Error(await res.text());
        const sub = await res.json();
        if (graphPathResult) {
          graphPathResult.innerHTML = `
            <strong>Subgraph around ${escapeHtml(center)}:</strong><br>
            <span class="badge badge-info">${sub.nodes.length} Nodes</span>
            <span class="badge badge-success">${sub.edges.length} Edges</span>
          `;
        }
      } catch (err) {
        if (graphPathResult) graphPathResult.innerHTML = `<span class="text-danger">Subgraph error: ${escapeHtml(err.message)}</span>`;
      } finally {
        btnExtractSubgraph.disabled = false;
      }
    });
  }

  if (btnSyncCampaignsGraph) {
    btnSyncCampaignsGraph.addEventListener('click', async () => {
      btnSyncCampaignsGraph.disabled = true;
      btnSyncCampaignsGraph.textContent = 'Syncing...';
      try {
        const res = await fetch('/api/graph/sync/campaigns', { method: 'POST' });
        if (!res.ok) throw new Error(await res.text());
        const result = await res.json();
        alert(`Graph synchronized successfully! (${result.campaigns_synced} campaigns, ${result.nodes_created} nodes created, ${result.edges_created} edges created)`);
        await loadGraphData();
      } catch (err) {
        alert('Campaign sync failed: ' + err.message);
      } finally {
        btnSyncCampaignsGraph.disabled = false;
        btnSyncCampaignsGraph.textContent = '🔄 Sync from Campaigns';
      }
    });
  }

  // Wire Daemon Cycle Modal & Form
  if (btnOpenDaemonModal && daemonModal) {
    btnOpenDaemonModal.addEventListener('click', () => { daemonModal.style.display = 'flex'; });
  }
  if (btnCloseDaemonModal && daemonModal) {
    btnCloseDaemonModal.addEventListener('click', () => { daemonModal.style.display = 'none'; });
  }
  if (btnCancelDaemon && daemonModal) {
    btnCancelDaemon.addEventListener('click', () => { daemonModal.style.display = 'none'; });
  }
  if (launchDaemonForm) {
    launchDaemonForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('btn-submit-daemon');
      const topic = document.getElementById('daemon-topic-input').value.trim();
      const context = document.getElementById('daemon-context-input').value.trim();
      const rounds = parseInt(document.getElementById('daemon-rounds-input').value || '2', 10);

      submitBtn.disabled = true;
      submitBtn.textContent = '🔬 Executing Research Cycle...';

      try {
        const res = await fetch('/api/research/daemon/cycle', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic, context, rounds })
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        daemonModal.style.display = 'none';
        launchDaemonForm.reset();
        await loadPapers();
        await loadGraphStats();
        if (data.paper_id) {
          await window.viewPaperDetails(data.paper_id);
        }
      } catch (err) {
        alert('Research cycle failed: ' + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '🔬 Launch Closed-Loop Cycle';
      }
    });
  }

  // Wire Vision Inspector Modal & Form
  if (btnOpenVisionModal && visionModal) {
    btnOpenVisionModal.addEventListener('click', () => { visionModal.style.display = 'flex'; });
  }
  if (btnCloseVisionModal && visionModal) {
    btnCloseVisionModal.addEventListener('click', () => { visionModal.style.display = 'none'; });
  }
  if (btnCancelVision && visionModal) {
    btnCancelVision.addEventListener('click', () => { visionModal.style.display = 'none'; });
  }
  if (inspectVisionForm) {
    inspectVisionForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('btn-submit-vision');
      const artifactName = document.getElementById('vision-artifact-name').value.trim();
      const artifactType = document.getElementById('vision-artifact-type').value;
      const expectedStr = document.getElementById('vision-expected-elements').value.trim();
      const expectedElements = expectedStr ? expectedStr.split(',').map(s => s.trim()).filter(Boolean) : [];
      const imageData = document.getElementById('vision-image-payload').value.trim();

      submitBtn.disabled = true;
      submitBtn.textContent = '👁️ Inspecting...';

      try {
        const res = await fetch('/api/multimodal/inspect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            artifact_name: artifactName,
            artifact_type: artifactType,
            expected_elements: expectedElements,
            image_data: imageData
          })
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        visionModal.style.display = 'none';
        inspectVisionForm.reset();
        await loadInspections();
        if (inspectionDetailCard) {
          if (inspectionDetailName) inspectionDetailName.textContent = data.artifact_name || artifactName;
          if (inspectionDetailBadge) {
            inspectionDetailBadge.className = data.passed ? 'badge badge-success' : 'badge badge-danger';
            inspectionDetailBadge.textContent = data.passed ? 'PASSED' : 'FAILED';
          }
          if (inspectionDetailReport) {
            inspectionDetailReport.textContent = data.analysis_report || 'Inspection completed.';
          }
          inspectionDetailCard.style.display = 'block';
        }
      } catch (err) {
        alert('Inspection failed: ' + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '👁️ Execute Inspection';
      }
    });
  }

  // ── Fleet & Self-Compilation Controller (Phase 11) ──
  const btnOpenRegisterFleetModal = document.getElementById('btn-open-register-fleet-modal');
  const btnCloseRegisterFleetModal = document.getElementById('btn-close-fleet-register-modal');
  const btnCancelFleetRegister = document.getElementById('btn-cancel-fleet-register');
  const fleetRegisterModal = document.getElementById('fleet-register-modal');
  const registerFleetForm = document.getElementById('register-fleet-form');

  const btnOpenImportFleetModal = document.getElementById('btn-open-import-fleet-modal');
  const btnCloseImportFleetModal = document.getElementById('btn-close-fleet-import-modal');
  const btnCancelFleetImport = document.getElementById('btn-cancel-fleet-import');
  const fleetImportModal = document.getElementById('fleet-import-modal');
  const importFleetForm = document.getElementById('import-fleet-form');

  const btnRefreshFleet = document.getElementById('btn-refresh-fleet');
  const btnExportFleetDelta = document.getElementById('btn-export-fleet-delta');
  const btnCompileBuiltinKernels = document.getElementById('btn-compile-builtin-kernels');

  async function loadFleetData() {
    await Promise.all([
      loadCompilers(),
      loadExtensions(),
      loadKernelExperiments(),
      loadModelMerges(),
      loadFleetNodes(),
      loadFleetLogs()
    ]);
  }

  async function loadCompilers() {
    const summaryEl = document.getElementById('compiler-detection-summary');
    const statEl = document.getElementById('stat-host-compilers');
    const subEl = document.getElementById('stat-host-compiler-sub');
    if (!summaryEl) return;

    try {
      const res = await fetch('/api/compiler/compilers');
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const compilers = data.compilers || {};
      const foundCount = Object.keys(compilers).length;

      if (statEl) {
        statEl.textContent = foundCount > 0 ? `${foundCount} Toolchains` : 'Python Fallback';
        statEl.style.color = foundCount > 0 ? '#22c55e' : '#f59e0b';
      }
      if (subEl) {
        subEl.textContent = foundCount > 0 ? Object.keys(compilers).join(', ') : 'Zero-overhead Python Fallback';
      }

      const allCandidates = [
        { id: 'clang', label: 'LLVM Clang', lang: 'C/C++' },
        { id: 'gcc', label: 'GNU GCC', lang: 'C/C++' },
        { id: 'cl', label: 'MSVC cl.exe', lang: 'C/C++' },
        { id: 'rustc', label: 'Rust rustc', lang: 'Rust' }
      ];

      summaryEl.innerHTML = allCandidates.map(c => {
        const path = compilers[c.id];
        const isDetected = Boolean(path);
        return `
          <div class="compiler-card ${isDetected ? 'detected' : 'absent'}">
            <div>
              <div class="compiler-name">${c.label}</div>
              <div class="compiler-sub" title="${path || 'Not detected on PATH'}">${path || 'Fallback Available'}</div>
            </div>
            <span class="badge ${isDetected ? 'badge-success' : 'badge-warning'}">
              ${isDetected ? 'ACTIVE' : 'FALLBACK'}
            </span>
          </div>
        `;
      }).join('');
    } catch (err) {
      summaryEl.innerHTML = `<div class="error-msg">Failed to query compilers: ${escapeHtml(err.message)}</div>`;
    }
  }

  async function loadExtensions() {
    const tbody = document.getElementById('extensions-tbody');
    const statEl = document.getElementById('stat-compiled-kernels');
    if (!tbody) return;

    try {
      const res = await fetch('/api/compiler/extensions');
      if (!res.ok) throw new Error(await res.text());
      const exts = await res.json();

      if (statEl) statEl.textContent = exts.length;

      if (!exts || exts.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-muted); padding:20px;">No compiled extensions loaded. Click 'Compile Built-in Kernels' to register.</td></tr>`;
        return;
      }

      tbody.innerHTML = exts.map(e => {
        const isNative = e.status === 'compiled';
        const speedupClass = isNative ? 'speedup-pill' : 'speedup-pill fallback';
        return `
          <tr>
            <td style="font-family:var(--font-mono); font-weight:600; color:var(--text-main);">${escapeHtml(e.name)}</td>
            <td><span class="badge badge-info">${escapeHtml(e.language.toUpperCase())}</span></td>
            <td><span class="badge ${isNative ? 'badge-success' : 'badge-warning'}">${escapeHtml(e.status)}</span></td>
            <td style="font-size:0.85rem; color:var(--text-muted);">${escapeHtml(e.compiler_used || 'Pure Python')}</td>
            <td><span class="${speedupClass}">${e.speedup_factor}x</span></td>
            <td style="font-size:0.8rem; color:var(--text-muted);">${formatDate(e.created_at)}</td>
            <td>
              <button class="btn btn-sm btn-secondary" onclick="window.benchmarkKernel('${escapeHtml(e.name)}')">⚡ Benchmark</button>
            </td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="7" class="error-msg">Error loading extensions: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  window.benchmarkKernel = async function(kernelName) {
    try {
      const res = await fetch(`/api/compiler/benchmark/${encodeURIComponent(kernelName)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ iterations: 500 })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      alert(`Benchmark Result for ${data.kernel}:\n\nSpeedup Factor: ${data.speedup_factor}x\nBaseline: ${data.baseline_ms} ms\nActive Kernel: ${data.active_ms} ms\nNative Shared Lib: ${data.is_native ? 'Yes' : 'No (Python Fallback)'}`);
      loadExtensions();
    } catch (err) {
      alert(`Benchmark failed: ${err.message}`);
    }
  };

  async function loadFleetNodes() {
    const grid = document.getElementById('fleet-nodes-grid');
    const statEl = document.getElementById('stat-fleet-nodes');
    if (!grid) return;

    try {
      const res = await fetch('/api/fleet/nodes');
      if (!res.ok) throw new Error(await res.text());
      const nodes = await res.json();

      if (statEl) statEl.textContent = nodes.length;

      if (!nodes || nodes.length === 0) {
        grid.innerHTML = `<div style="grid-column: 1 / -1; text-align:center; color:var(--text-muted); padding:24px;">No fleet nodes discovered.</div>`;
        return;
      }

      grid.innerHTML = nodes.map(n => {
        const isSelf = n.metadata && n.metadata.is_self;
        const statusBadgeClass = n.status === 'online' ? 'badge-success' : (n.status === 'syncing' ? 'badge-info' : 'badge-warning');
        return `
          <div class="fleet-node-card ${isSelf ? 'self-node' : 'peer-node'}">
            <div class="node-card-header">
              <div class="node-card-title">
                ${isSelf ? '🏠' : '🖥️'} ${escapeHtml(n.hostname)}
                ${isSelf ? '<span class="badge badge-info" style="font-size:0.7rem;">THIS HOST</span>' : ''}
              </div>
              <span class="badge ${statusBadgeClass}">${escapeHtml(n.status.toUpperCase())}</span>
            </div>
            <div class="node-card-endpoint">${escapeHtml(n.address)}:${n.port}</div>
            <div class="node-card-details">
              <div><strong>GPU:</strong> ${escapeHtml(n.gpu_device || 'CPU-only')}</div>
              <div><strong>VRAM:</strong> ${n.vram_gb > 0 ? n.vram_gb + ' GB' : 'N/A'}</div>
              <div><strong>Heartbeat:</strong> ${formatDate(n.last_heartbeat)}</div>
              ${n.sync_cursor ? `<div><strong>Sync Cursor:</strong> ${formatDate(n.sync_cursor)}</div>` : ''}
            </div>
            <div class="node-card-actions">
              <button class="btn btn-sm btn-secondary" onclick="window.pingFleetNode('${escapeHtml(n.id)}')">📡 Ping</button>
              ${!isSelf ? `
                <button class="btn btn-sm btn-primary" onclick="window.syncFleetNode('${escapeHtml(n.id)}')">🔄 Sync Delta</button>
                <button class="btn btn-sm btn-danger" onclick="window.removeFleetNode('${escapeHtml(n.id)}')">🗑️</button>
              ` : ''}
            </div>
          </div>
        `;
      }).join('');
    } catch (err) {
      grid.innerHTML = `<div class="error-msg">Failed to query fleet nodes: ${escapeHtml(err.message)}</div>`;
    }
  }

  window.pingFleetNode = async function(nodeId) {
    try {
      const res = await fetch(`/api/fleet/nodes/${encodeURIComponent(nodeId)}/ping`, { method: 'POST' });
      if (!res.ok) throw new Error(await res.text());
      loadFleetNodes();
    } catch (err) {
      alert(`Ping failed: ${err.message}`);
    }
  };

  window.syncFleetNode = async function(nodeId) {
    try {
      const res = await fetch(`/api/fleet/sync/${encodeURIComponent(nodeId)}`, { method: 'POST' });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      alert(`Sync completed: ${data.message || 'Success'} (Status: ${data.status})`);
      loadFleetData();
    } catch (err) {
      alert(`Delta sync failed: ${err.message}`);
    }
  };

  window.removeFleetNode = async function(nodeId) {
    if (!confirm('Are you sure you want to remove this peer node from the cluster?')) return;
    try {
      const res = await fetch(`/api/fleet/nodes/${encodeURIComponent(nodeId)}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(await res.text());
      loadFleetNodes();
    } catch (err) {
      alert(`Remove failed: ${err.message}`);
    }
  };

  async function loadFleetLogs() {
    const tbody = document.getElementById('fleet-logs-tbody');
    const statEl = document.getElementById('stat-sync-events');
    if (!tbody) return;

    try {
      const res = await fetch('/api/fleet/logs?limit=50');
      if (!res.ok) throw new Error(await res.text());
      const logs = await res.json();

      if (statEl) statEl.textContent = logs.length;

      if (!logs || logs.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color:var(--text-muted); padding:20px;">No sync events recorded yet.</td></tr>`;
        return;
      }

      tbody.innerHTML = logs.map(l => {
        const isSuccess = l.status === 'success';
        return `
          <tr>
            <td style="font-family:var(--font-mono); font-size:0.8rem; color:var(--text-muted);">${escapeHtml(l.id)}</td>
            <td style="font-weight:600;">${escapeHtml(l.node_id)}</td>
            <td><span class="badge badge-info">${escapeHtml(l.direction.toUpperCase())}</span></td>
            <td style="font-size:0.85rem;">${escapeHtml(l.entity_type)}</td>
            <td style="font-weight:600; color:var(--primary);">${l.items_synced} items</td>
            <td><span class="badge ${isSuccess ? 'badge-success' : 'badge-danger'}">${escapeHtml(l.status)}</span></td>
            <td style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted);">${escapeHtml((l.checksum || '').slice(0, 12))}...</td>
            <td style="font-size:0.8rem; color:var(--text-muted);">${formatDate(l.created_at)}</td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="8" class="error-msg">Error loading logs: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  // Built-in hot-path kernels
  const C_DOT_PRODUCT_SRC = `#include <stddef.h>\\n#ifdef _WIN32\\n#define EXPORT __declspec(dllexport)\\n#else\\n#define EXPORT\\n#endif\\nEXPORT float dot_product(const float* a, const float* b, size_t n) {\\n    float sum = 0.0f;\\n    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];\\n    return sum;\\n}`;

  if (btnCompileBuiltinKernels) {
    btnCompileBuiltinKernels.addEventListener('click', async () => {
      btnCompileBuiltinKernels.disabled = true;
      btnCompileBuiltinKernels.textContent = '⚡ Synthesizing & Compiling...';
      try {
        const res = await fetch('/api/compiler/compile', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: 'vector_dot_product',
            language: 'c',
            source_code: C_DOT_PRODUCT_SRC,
            metadata: { built_in: true }
          })
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        alert(`Kernel '${data.name}' registered with status: ${data.status} (Compiler: ${data.compiler_used || 'Pure Python Fallback'})`);
        loadExtensions();
      } catch (err) {
        alert(`Compilation failed: ${err.message}`);
      } finally {
        btnCompileBuiltinKernels.disabled = false;
        btnCompileBuiltinKernels.textContent = '⚡ Compile Built-in Kernels';
      }
    });
  }

  if (btnRefreshFleet) {
    btnRefreshFleet.addEventListener('click', () => { loadFleetData(); });
  }

  if (btnExportFleetDelta) {
    btnExportFleetDelta.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/fleet/export-delta');
        if (!res.ok) throw new Error(await res.text());
        const bundle = await res.json();
        const jsonStr = JSON.stringify(bundle, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `genesis-fleet-delta-${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } catch (err) {
        alert(`Export failed: ${err.message}`);
      }
    });
  }

  // Modal Wiring: Register Fleet Node
  if (btnOpenRegisterFleetModal && fleetRegisterModal) {
    btnOpenRegisterFleetModal.addEventListener('click', () => { fleetRegisterModal.style.display = 'flex'; });
  }
  if (btnCloseRegisterFleetModal && fleetRegisterModal) {
    btnCloseRegisterFleetModal.addEventListener('click', () => { fleetRegisterModal.style.display = 'none'; });
  }
  if (btnCancelFleetRegister && fleetRegisterModal) {
    btnCancelFleetRegister.addEventListener('click', () => { fleetRegisterModal.style.display = 'none'; });
  }
  if (registerFleetForm) {
    registerFleetForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const hostname = document.getElementById('fleet-hostname').value.trim();
      const address = document.getElementById('fleet-address').value.trim();
      const port = parseInt(document.getElementById('fleet-port').value, 10) || 8000;
      const gpu = document.getElementById('fleet-gpu').value.trim();
      const vram = parseFloat(document.getElementById('fleet-vram').value) || 0.0;

      try {
        const res = await fetch('/api/fleet/nodes', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            hostname: hostname,
            address: address,
            port: port,
            gpu_device: gpu,
            vram_gb: vram
          })
        });
        if (!res.ok) throw new Error(await res.text());
        fleetRegisterModal.style.display = 'none';
        registerFleetForm.reset();
        loadFleetNodes();
      } catch (err) {
        alert(`Failed to register peer node: ${err.message}`);
      }
    });
  }

  // Modal Wiring: Import Fleet Delta
  if (btnOpenImportFleetModal && fleetImportModal) {
    btnOpenImportFleetModal.addEventListener('click', () => { fleetImportModal.style.display = 'flex'; });
  }
  if (btnCloseImportFleetModal && fleetImportModal) {
    btnCloseImportFleetModal.addEventListener('click', () => { fleetImportModal.style.display = 'none'; });
  }
  if (btnCancelFleetImport && fleetImportModal) {
    btnCancelFleetImport.addEventListener('click', () => { fleetImportModal.style.display = 'none'; });
  }
  if (importFleetForm) {
    importFleetForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const sourceNodeId = document.getElementById('fleet-import-source-node').value.trim();
      const payloadStr = document.getElementById('fleet-import-payload').value.trim();

      try {
        const bundle = JSON.parse(payloadStr);
        const res = await fetch('/api/fleet/import-delta', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            bundle: bundle,
            source_node_id: sourceNodeId || 'manual-offline-import'
          })
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        alert(`Delta import finished: ${data.message || 'Done'} (Synced: ${data.items_synced} entities)`);
        fleetImportModal.style.display = 'none';
        importFleetForm.reset();
        loadFleetData();
      } catch (err) {
        alert(`Import failed: ${err.message}`);
      }
    });
  }

  // ── Helper: Format Date ──
  function formatDate(isoStr) {
    if (!isoStr) return 'N/A';
    try {
      const d = new Date(isoStr);
      return isNaN(d.getTime()) ? String(isoStr).slice(0, 19).replace('T', ' ') : d.toLocaleString();
    } catch {
      return String(isoStr).slice(0, 19).replace('T', ' ');
    }
  }

  // ── Phase 12: Autonomous Kernel Synthesis Controller ──
  let cachedKernelExperiments = [];

  async function loadKernelExperiments() {
    const tbody = document.getElementById('kernel-experiments-tbody');
    const statEl = document.getElementById('stat-kernel-experiments');
    if (!tbody) return;

    try {
      const res = await fetch('/api/kernel-synthesis/experiments');
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const exps = Array.isArray(data) ? data : (data.experiments || []);
      cachedKernelExperiments = exps;

      if (statEl) statEl.textContent = exps.length;

      if (!exps || exps.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:var(--text-muted); padding:20px;">No kernel synthesis experiments recorded yet. Click 'Run Synthesis Loop' to launch.</td></tr>`;
        return;
      }

      tbody.innerHTML = exps.map(e => {
        const isSuccess = e.status === 'success' || e.status === 'repair_succeeded';
        const statusBadge = isSuccess ? 'badge-success' : (e.status === 'fallback_python' ? 'badge-warning' : 'badge-danger');
        const numCheck = e.numerical_check_passed ? '<span class="badge badge-success">PASSED</span>' : '<span class="badge badge-danger">FAILED</span>';
        const speedup = typeof e.speedup_factor === 'number' ? `${e.speedup_factor.toFixed(2)}x` : '1.00x';
        const maxErr = typeof e.max_abs_error === 'number' ? e.max_abs_error.toExponential(2) : '0.00e+0';

        return `
          <tr>
            <td style="font-family:var(--font-mono); font-size:0.8rem; color:var(--text-muted);">${escapeHtml(e.id)}</td>
            <td style="font-weight:600; font-family:var(--font-mono);">${escapeHtml(e.operation_type)}</td>
            <td><span class="badge ${statusBadge}">${escapeHtml(e.status)}</span></td>
            <td style="font-weight:600; color:var(--primary);">${escapeHtml(speedup)}</td>
            <td style="font-family:var(--font-mono); font-size:0.85rem;">${escapeHtml(maxErr)}</td>
            <td>${numCheck}</td>
            <td style="text-align:center; font-weight:600;">${e.iteration_count || 1}</td>
            <td style="font-size:0.8rem; color:var(--text-muted);">${formatDate(e.created_at)}</td>
            <td>
              <button class="btn btn-sm btn-secondary" onclick="window.viewKernelSource('${escapeHtml(e.id)}')">🔍 View C</button>
            </td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="9" class="error-msg">Error loading kernel experiments: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  window.viewKernelSource = function(expId) {
    const exp = cachedKernelExperiments.find(x => x.id === expId);
    const modal = document.getElementById('kernel-code-modal');
    const title = document.getElementById('kernel-code-title');
    const pre = document.getElementById('kernel-code-pre');
    if (!modal || !pre) return;

    if (exp) {
      if (title) title.textContent = `Generated C Kernel: ${exp.operation_type} (${exp.status})`;
      pre.textContent = exp.c_source || '// No C source available for this run.';
    } else {
      if (title) title.textContent = 'Kernel Source';
      pre.textContent = `Experiment ${expId} not found in cache.`;
    }
    modal.style.display = 'flex';
  };

  const kernelCodeModal = document.getElementById('kernel-code-modal');
  const btnCloseKernelCodeModal = document.getElementById('btn-close-kernel-code-modal');
  const btnCloseKernelCodeBtn = document.getElementById('btn-close-kernel-code-btn');
  if (btnCloseKernelCodeModal && kernelCodeModal) {
    btnCloseKernelCodeModal.addEventListener('click', () => { kernelCodeModal.style.display = 'none'; });
  }
  if (btnCloseKernelCodeBtn && kernelCodeModal) {
    btnCloseKernelCodeBtn.addEventListener('click', () => { kernelCodeModal.style.display = 'none'; });
  }

  // Kernel Synthesis Modal Wiring
  const kernelModal = document.getElementById('kernel-synthesis-modal');
  const btnOpenKernelModal = document.getElementById('btn-open-kernel-synthesis-modal');
  const btnOpenKernelModalAlt = document.getElementById('btn-open-kernel-synthesis-modal-alt');
  const btnCloseKernelModal = document.getElementById('btn-close-kernel-modal');
  const btnCancelKernelModal = document.getElementById('btn-cancel-kernel-modal');
  const kernelForm = document.getElementById('kernel-synthesis-form');

  [btnOpenKernelModal, btnOpenKernelModalAlt].forEach(btn => {
    if (btn && kernelModal) {
      btn.addEventListener('click', () => { kernelModal.style.display = 'flex'; });
    }
  });
  [btnCloseKernelModal, btnCancelKernelModal].forEach(btn => {
    if (btn && kernelModal) {
      btn.addEventListener('click', () => { kernelModal.style.display = 'none'; });
    }
  });

  if (kernelForm) {
    kernelForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const op = document.getElementById('kernel-op-select').value;
      const iters = parseInt(document.getElementById('kernel-iterations').value, 10) || 3;
      const tol = parseFloat(document.getElementById('kernel-tolerance').value) || 0.0001;
      const submitBtn = document.getElementById('btn-submit-kernel-modal');

      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = '⚡ Synthesizing & Verifying...';
      }

      try {
        const res = await fetch('/api/kernel-synthesis/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            operation_type: op,
            max_iterations: iters,
            tolerance: tol
          })
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        alert(`Kernel synthesis finished!\nOperation: ${data.operation_type}\nStatus: ${data.status}\nSpeedup: ${data.speedup_factor}x\nNumerical check: ${data.numerical_check_passed ? 'PASSED' : 'FAILED'}`);
        if (kernelModal) kernelModal.style.display = 'none';
        kernelForm.reset();
        await loadKernelExperiments();
      } catch (err) {
        alert(`Synthesis failed: ${err.message}`);
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = '⚡ Launch Synthesis Loop';
        }
      }
    });
  }

  // ── Phase 12: Model Parameter Merging Controller ──
  async function loadModelMerges() {
    const tbody = document.getElementById('model-merges-tbody');
    const statEl = document.getElementById('stat-model-merges');
    if (!tbody) return;

    try {
      const res = await fetch('/api/merging/merges');
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const merges = Array.isArray(data) ? data : (data.merges || []);

      if (statEl) statEl.textContent = merges.length;

      if (!merges || merges.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color:var(--text-muted); padding:20px;">No model parameter merges registered yet. Click 'Execute Model Merge' to start.</td></tr>`;
        return;
      }

      tbody.innerHTML = merges.map(m => {
        const densityPct = typeof m.density === 'number' ? `${(m.density * 100).toFixed(0)}%` : '100%';
        const paramCount = m.metrics && m.metrics.merged_parameters_count ? m.metrics.merged_parameters_count : (m.metrics && m.metrics.parameters_merged ? m.metrics.parameters_merged : 'N/A');

        return `
          <tr>
            <td style="font-family:var(--font-mono); font-size:0.8rem; color:var(--text-muted);">${escapeHtml(m.id)}</td>
            <td style="font-weight:600;">${escapeHtml(m.name)}</td>
            <td><code>${escapeHtml(m.base_model || 'default')}</code></td>
            <td><span class="badge badge-info">${escapeHtml(m.method ? m.method.toUpperCase() : 'LINEAR')}</span></td>
            <td style="font-weight:600; color:var(--primary);">${escapeHtml(densityPct)}</td>
            <td style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted);" title="${escapeHtml(m.output_path || '')}">${escapeHtml((m.output_path || '').slice(0, 24))}...</td>
            <td style="font-weight:600;">${escapeHtml(String(paramCount))}</td>
            <td style="font-size:0.8rem; color:var(--text-muted);">${formatDate(m.created_at)}</td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="8" class="error-msg">Error loading model merges: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  // Model Merge Modal Wiring
  const mergeModal = document.getElementById('model-merge-modal');
  const btnOpenMergeModal = document.getElementById('btn-open-model-merge-modal');
  const btnOpenMergeModalAlt = document.getElementById('btn-open-model-merge-modal-alt');
  const btnCloseMergeModal = document.getElementById('btn-close-merge-modal');
  const btnCancelMergeModal = document.getElementById('btn-cancel-merge-modal');
  const mergeForm = document.getElementById('model-merge-form');
  const mergeDensitySlider = document.getElementById('merge-density');
  const mergeDensityVal = document.getElementById('merge-density-val');

  if (mergeDensitySlider && mergeDensityVal) {
    mergeDensitySlider.addEventListener('input', () => {
      mergeDensityVal.textContent = parseFloat(mergeDensitySlider.value).toFixed(2);
    });
  }

  [btnOpenMergeModal, btnOpenMergeModalAlt].forEach(btn => {
    if (btn && mergeModal) {
      btn.addEventListener('click', () => { mergeModal.style.display = 'flex'; });
    }
  });
  [btnCloseMergeModal, btnCancelMergeModal].forEach(btn => {
    if (btn && mergeModal) {
      btn.addEventListener('click', () => { mergeModal.style.display = 'none'; });
    }
  });

  if (mergeForm) {
    mergeForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const name = document.getElementById('merge-recipe-name').value.trim();
      const method = document.getElementById('merge-method-select').value;
      const baseModel = document.getElementById('merge-base-model').value.trim();
      const density = parseFloat(document.getElementById('merge-density').value) || 0.8;
      const weightsRaw = document.getElementById('merge-weights').value.trim();

      let weightsList = [0.5, 0.5];
      if (weightsRaw) {
        try {
          if (weightsRaw.startsWith('[')) {
            weightsList = JSON.parse(weightsRaw);
          } else {
            weightsList = weightsRaw.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
          }
        } catch {
          weightsList = [0.5, 0.5];
        }
      }

      const submitBtn = document.getElementById('btn-submit-merge-modal');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = '🔀 Merging Model Parameters...';
      }

      try {
        const res = await fetch('/api/merging/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: name,
            method: method,
            base_model: baseModel,
            density: density,
            weights: weightsList,
            adapter_paths: ['data/adapters/adapter_a.pt', 'data/adapters/adapter_b.pt']
          })
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        alert(`Model parameter merge executed successfully!\nMerge ID: ${data.id}\nMethod: ${data.method.toUpperCase()}\nOutput Path: ${data.output_path}`);
        if (mergeModal) mergeModal.style.display = 'none';
        mergeForm.reset();
        await loadModelMerges();
      } catch (err) {
        alert(`Merge execution failed: ${err.message}`);
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = '🔀 Execute Merge';
        }
      }
    });
  }

  // ── Phase 12: Human-in-the-Loop Operator Steering Controller ──
  async function loadSteeringDirectives() {
    const tbody = document.getElementById('steering-directives-tbody');
    if (!tbody) return;

    try {
      const res = await fetch('/api/steering/directives');
      if (!res.ok) throw new Error(await res.text());
      const directives = await res.json();

      if (!directives || directives.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No steering directives active.</td></tr>`;
        return;
      }

      tbody.innerHTML = directives.map(d => {
        const isApplied = d.status === 'applied';
        const statusBadge = isApplied ? 'badge-success' : 'badge-warning';
        const actionBadge = d.action === 'abort' ? 'badge-danger' : (d.action === 'pause' ? 'badge-warning' : 'badge-info');

        let payloadStr = '';
        if (d.payload) {
          if (typeof d.payload === 'string') {
            payloadStr = d.payload;
          } else if (d.payload.notes || d.payload.guidance || d.payload.critique) {
            payloadStr = d.payload.notes || d.payload.guidance || d.payload.critique;
          } else {
            payloadStr = JSON.stringify(d.payload);
          }
        }

        return `
          <tr>
            <td style="font-family:var(--font-mono); font-size:0.8rem; color:var(--text-muted);">${escapeHtml(d.id)}</td>
            <td><span class="badge ${actionBadge}">${escapeHtml(d.action ? d.action.toUpperCase() : 'DIRECTIVE')}</span></td>
            <td><code>${escapeHtml(d.target_subsystem || 'general')}</code></td>
            <td><span class="badge ${statusBadge}">${escapeHtml(d.status ? d.status.toUpperCase() : 'RECEIVED')}</span></td>
            <td style="font-size:0.85rem; max-width:280px; word-break:break-word;">${escapeHtml(payloadStr.slice(0, 100))}</td>
            <td style="font-size:0.8rem; color:var(--text-muted);">${formatDate(d.created_at)}</td>
            <td>
              ${!isApplied ? `<button class="btn btn-sm btn-primary" onclick="window.applySteeringDirective('${escapeHtml(d.id)}')">✓ Mark Applied</button>` : `<span style="color:var(--text-muted); font-size:0.8rem;">Applied</span>`}
            </td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="7" class="error-msg">Error loading steering directives: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  window.applySteeringDirective = async function(directiveId) {
    try {
      const res = await fetch(`/api/steering/directives/${encodeURIComponent(directiveId)}/apply`, {
        method: 'POST'
      });
      if (!res.ok) throw new Error(await res.text());
      await loadSteeringDirectives();
    } catch (err) {
      alert(`Failed to apply directive: ${err.message}`);
    }
  };

  // Steering Directive Modal Wiring
  const steeringModal = document.getElementById('steering-modal');
  const btnOpenSteeringModal = document.getElementById('btn-open-steering-modal');
  const btnCloseSteeringModal = document.getElementById('btn-close-steering-modal');
  const btnCancelSteeringModal = document.getElementById('btn-cancel-steering-modal');
  const steeringForm = document.getElementById('steering-directive-form');

  if (btnOpenSteeringModal && steeringModal) {
    btnOpenSteeringModal.addEventListener('click', () => { steeringModal.style.display = 'flex'; });
  }
  [btnCloseSteeringModal, btnCancelSteeringModal].forEach(btn => {
    if (btn && steeringModal) {
      btn.addEventListener('click', () => { steeringModal.style.display = 'none'; });
    }
  });

  if (steeringForm) {
    steeringForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const action = document.getElementById('steering-action-select').value;
      const targetSubsystem = document.getElementById('steering-target-select').value;
      const notes = document.getElementById('steering-notes').value.trim();

      const submitBtn = document.getElementById('btn-submit-steering-modal');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = '🎯 Transmitting...';
      }

      try {
        const res = await fetch('/api/steering/directives', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: action,
            target_subsystem: targetSubsystem,
            payload: { notes: notes, timestamp: new Date().toISOString() }
          })
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        alert(`Steering directive '${data.action.toUpperCase()}' dispatched successfully!`);
        if (steeringModal) steeringModal.style.display = 'none';
        steeringForm.reset();
        await loadSteeringDirectives();
      } catch (err) {
        alert(`Failed to issue steering directive: ${err.message}`);
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = '🎯 Transmit Directive';
        }
      }
    });
  // ── Phase 13: Continual Online Learning & Dynamic Experience Replay ──
  async function loadReplayBuffer() {
    const tbody = document.getElementById('replay-samples-tbody');
    const totalCount = document.getElementById('replay-total-count');
    const capacitySub = document.getElementById('replay-capacity-sub');

    try {
      // 1. Load buffer stats
      const statsRes = await fetch('/api/continual/replay/stats');
      if (statsRes.ok) {
        const stats = await statsRes.json();
        if (totalCount) totalCount.textContent = stats.total_samples || 0;
        if (capacitySub && stats.capacity_per_domain) {
          capacitySub.textContent = `Capacity: ${stats.capacity_per_domain} / domain`;
        }
      }

      // 2. Load samples
      const res = await fetch('/api/continual/replay/samples?limit=25');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const samples = data.samples || [];

      if (!tbody) return;
      if (samples.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No replay samples in buffer.</td></tr>';
        return;
      }

      tbody.innerHTML = samples.map(s => {
        const domainBadge = {
          reasoning: '<span class="badge" style="background:#8b5cf6;">reasoning</span>',
          coding: '<span class="badge" style="background:#06b6d4;">coding</span>',
          tools: '<span class="badge" style="background:#f59e0b;">tools</span>',
          general: '<span class="badge" style="background:#10b981;">general</span>',
        }[s.task_domain] || `<span class="badge">${escapeHtml(s.task_domain)}</span>`;

        const inPreview = s.sample_input.length > 50 ? s.sample_input.slice(0, 50) + '...' : s.sample_input;
        const outPreview = s.sample_target.length > 50 ? s.sample_target.slice(0, 50) + '...' : s.sample_target;
        const created = s.created_at ? s.created_at.slice(0, 19).replace('T', ' ') : '--';

        return `
          <tr>
            <td>${domainBadge}</td>
            <td><strong>${s.priority_score.toFixed(2)}</strong></td>
            <td>${s.usage_count}</td>
            <td title="${escapeHtml(s.sample_input)}">${escapeHtml(inPreview)}</td>
            <td title="${escapeHtml(s.sample_target)}">${escapeHtml(outPreview)}</td>
            <td style="font-size:0.75rem; color:var(--text-muted);">${escapeHtml(created)}</td>
            <td>
              <button class="btn btn-sm btn-secondary" onclick="window.deleteReplaySample('${escapeHtml(s.id)}')">🗑️</button>
            </td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      if (tbody) {
        tbody.innerHTML = `<tr><td colspan="7" class="error-msg">Error loading replay buffer: ${escapeHtml(err.message)}</td></tr>`;
      }
    }
  }

  async function loadEWC() {
    const tbody = document.getElementById('ewc-checkpoints-tbody');
    const countEl = document.getElementById('ewc-checkpoints-count');

    try {
      const res = await fetch('/api/continual/ewc/checkpoints');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const records = data.records || [];

      if (countEl) countEl.textContent = records.length;
      if (!tbody) return;

      if (records.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No EWC parameter checkpoints consolidated yet.</td></tr>';
        return;
      }

      tbody.innerHTML = records.map(r => {
        const created = r.created_at ? r.created_at.slice(0, 19).replace('T', ' ') : '--';
        return `
          <tr>
            <td><strong>${escapeHtml(r.task_name)}</strong></td>
            <td><code>${escapeHtml(r.experiment_id)}</code></td>
            <td>&lambda; = ${r.importance_lambda}</td>
            <td>${r.parameter_count} weights</td>
            <td style="font-size:0.75rem; color:var(--text-muted);">${escapeHtml(created)}</td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      if (tbody) {
        tbody.innerHTML = `<tr><td colspan="5" class="error-msg">Error loading EWC checkpoints: ${escapeHtml(err.message)}</td></tr>`;
      }
    }
  }

  window.deleteReplaySample = async function(sampleId) {
    if (!confirm('Remove this sample from the replay buffer?')) return;
    try {
      const res = await fetch(`/api/continual/replay/samples/${encodeURIComponent(sampleId)}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await loadReplayBuffer();
    } catch (err) {
      alert(`Failed to delete sample: ${err.message}`);
    }
  };

  // Replay Sample Modal Wiring
  const replayModal = document.getElementById('replay-sample-modal');
  const btnOpenReplayModal = document.getElementById('btn-open-add-replay-modal');
  const btnCloseReplayModal = document.getElementById('btn-close-replay-sample-modal');
  const btnCancelReplayModal = document.getElementById('btn-cancel-replay-sample-modal');
  const replayForm = document.getElementById('replay-sample-form');

  if (btnOpenReplayModal && replayModal) {
    btnOpenReplayModal.addEventListener('click', () => { replayModal.style.display = 'flex'; });
  }
  [btnCloseReplayModal, btnCancelReplayModal].forEach(btn => {
    if (btn && replayModal) {
      btn.addEventListener('click', () => { replayModal.style.display = 'none'; });
    }
  });

  if (replayForm) {
    replayForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const domain = document.getElementById('replay-domain-select').value;
      const priority = parseFloat(document.getElementById('replay-priority-input').value) || 1.0;
      const input = document.getElementById('replay-input-text').value.trim();
      const target = document.getElementById('replay-target-text').value.trim();
      const submitBtn = document.getElementById('btn-submit-replay-sample');

      try {
        if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = 'Saving...'; }
        const res = await fetch('/api/continual/replay/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            task_domain: domain,
            priority_score: priority,
            sample_input: input,
            sample_target: target
          }),
        });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `HTTP ${res.status}`);
        }
        if (replayModal) replayModal.style.display = 'none';
        replayForm.reset();
        await loadReplayBuffer();
      } catch (err) {
        alert(`Failed to add replay sample: ${err.message}`);
      } finally {
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = '📥 Add to Replay Buffer'; }
      }
    });
  }

  // Balanced Batch Sampling Modal Wiring
  const batchModal = document.getElementById('replay-batch-modal');
  const btnOpenBatchModal = document.getElementById('btn-open-sample-batch-modal');
  const btnCloseBatchModal = document.getElementById('btn-close-replay-batch-modal');
  const btnCloseBatchModalBtn = document.getElementById('btn-close-batch-modal-btn');
  const btnGenerateBatch = document.getElementById('btn-generate-batch');
  const batchResultsContainer = document.getElementById('batch-results-container');

  if (btnOpenBatchModal && batchModal) {
    btnOpenBatchModal.addEventListener('click', () => { batchModal.style.display = 'flex'; });
  }
  [btnCloseBatchModal, btnCloseBatchModalBtn].forEach(btn => {
    if (btn && batchModal) {
      btn.addEventListener('click', () => { batchModal.style.display = 'none'; });
    }
  });

  if (btnGenerateBatch && batchResultsContainer) {
    btnGenerateBatch.addEventListener('click', async () => {
      const batchSize = parseInt(document.getElementById('batch-size-input').value, 10) || 4;
      try {
        btnGenerateBatch.disabled = true;
        btnGenerateBatch.textContent = 'Sampling...';
        const res = await fetch(`/api/continual/replay/batch?batch_size=${batchSize}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const batch = data.batch || [];

        batchResultsContainer.style.display = 'block';
        if (batch.length === 0) {
          batchResultsContainer.innerHTML = '<p class="text-muted text-sm">No samples available in buffer to form a batch.</p>';
        } else {
          batchResultsContainer.innerHTML = batch.map(b => `
            <div style="background:var(--bg-card); border:1px solid var(--border-color); border-radius:6px; padding:10px; margin-bottom:8px;">
              <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span class="badge">${escapeHtml(b.task_domain)}</span>
                <span style="font-size:0.75rem; color:var(--text-muted);">Priority: ${b.priority_score.toFixed(2)} | Usage: ${b.usage_count}</span>
              </div>
              <p style="margin:2px 0; font-size:0.85rem;"><strong>Prompt:</strong> ${escapeHtml(b.sample_input)}</p>
              <p style="margin:2px 0; font-size:0.85rem; color:var(--text-muted);"><strong>Target:</strong> ${escapeHtml(b.sample_target)}</p>
            </div>
          `).join('');
        }
        await loadReplayBuffer();
      } catch (err) {
        alert(`Failed to sample rehearsal batch: ${err.message}`);
      } finally {
        btnGenerateBatch.disabled = false;
        btnGenerateBatch.textContent = '⚡ Sample Replay Batch';
      }
    });
  }

  // Refresh buttons
  const btnRefreshReplay = document.getElementById('btn-refresh-replay-buffer');
  if (btnRefreshReplay) btnRefreshReplay.addEventListener('click', () => loadReplayBuffer());
  const btnRefreshEWC = document.getElementById('btn-refresh-ewc-table');
  if (btnRefreshEWC) btnRefreshEWC.addEventListener('click', () => loadEWC());

  // ── Phase 13: Neuro-Symbolic Verification Engine ──
  async function loadSymbolicVerifications() {
    const tbody = document.getElementById('symbolic-verifications-tbody');
    try {
      const res = await fetch('/api/symbolic/verifications?limit=25');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const records = data.records || [];

      if (!tbody) return;
      if (records.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No formal verifications recorded yet.</td></tr>';
        return;
      }

      tbody.innerHTML = records.map(r => {
        let statusBadge = '<span class="badge badge-warning">inconclusive</span>';
        if (r.status === 'verified') statusBadge = '<span class="badge badge-success">✓ verified</span>';
        if (r.status === 'violated') statusBadge = '<span class="badge badge-danger">✗ violated</span>';

        const exprPreview = (r.expression_or_code || '').length > 45
          ? (r.expression_or_code || '').slice(0, 45) + '...'
          : (r.expression_or_code || '');
        const violations = (r.violations || []).join(', ') || 'None';
        const created = r.created_at ? r.created_at.slice(0, 19).replace('T', ' ') : '--';

        return `
          <tr>
            <td><code>${escapeHtml(r.id.slice(0, 12))}</code></td>
            <td><span class="badge">${escapeHtml(r.target_type)}</span></td>
            <td>${statusBadge}</td>
            <td title="${escapeHtml(r.expression_or_code || '')}"><code>${escapeHtml(exprPreview)}</code></td>
            <td style="font-size:0.8rem; color:${r.violations && r.violations.length > 0 ? 'var(--color-danger, #ef4444)' : 'var(--text-muted)'};">${escapeHtml(violations)}</td>
            <td style="font-size:0.8rem;">${r.verification_latency_ms.toFixed(1)} ms</td>
            <td style="font-size:0.75rem; color:var(--text-muted);">${escapeHtml(created)}</td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      if (tbody) {
        tbody.innerHTML = `<tr><td colspan="7" class="error-msg">Error loading verifications: ${escapeHtml(err.message)}</td></tr>`;
      }
    }
  }

  // Symbolic Verify Modal Wiring
  const verifyModal = document.getElementById('symbolic-verify-modal');
  const btnOpenVerifyModal = document.getElementById('btn-open-verify-modal');
  const btnCloseVerifyModal = document.getElementById('btn-close-symbolic-modal');
  const btnCancelVerifyModal = document.getElementById('btn-cancel-symbolic-modal');
  const verifyForm = document.getElementById('symbolic-verify-form');
  const verifyTypeSelect = document.getElementById('symbolic-type-select');
  const codeFields = document.getElementById('symbolic-code-fields');
  const mathFields = document.getElementById('symbolic-math-fields');
  const logicFields = document.getElementById('symbolic-logic-fields');
  const verdictBox = document.getElementById('symbolic-verify-verdict');

  if (btnOpenVerifyModal && verifyModal) {
    btnOpenVerifyModal.addEventListener('click', () => {
      verifyModal.style.display = 'flex';
      if (verdictBox) verdictBox.style.display = 'none';
    });
  }
  [btnCloseVerifyModal, btnCancelVerifyModal].forEach(btn => {
    if (btn && verifyModal) {
      btn.addEventListener('click', () => { verifyModal.style.display = 'none'; });
    }
  });

  if (verifyTypeSelect) {
    verifyTypeSelect.addEventListener('change', () => {
      const val = verifyTypeSelect.value;
      if (codeFields) codeFields.style.display = val === 'code' ? 'block' : 'none';
      if (mathFields) mathFields.style.display = val === 'math' ? 'block' : 'none';
      if (logicFields) logicFields.style.display = val === 'logic' ? 'block' : 'none';
      if (verdictBox) verdictBox.style.display = 'none';
    });
  }

  if (verifyForm) {
    verifyForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const targetType = verifyTypeSelect.value;
      const submitBtn = document.getElementById('btn-submit-symbolic-verify');

      try {
        if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = 'Verifying...'; }
        let endpoint = '/api/symbolic/verify/code';
        let payload = {};

        if (targetType === 'code') {
          endpoint = '/api/symbolic/verify/code';
          payload = {
            code: document.getElementById('symbolic-code-input').value,
            check_termination: true,
            check_safety: true,
            check_division_by_zero: true
          };
        } else if (targetType === 'math') {
          endpoint = '/api/symbolic/verify/math';
          payload = {
            expression: document.getElementById('symbolic-math-expr').value,
            domain_min: parseFloat(document.getElementById('symbolic-math-min').value) || 0.0,
            domain_max: parseFloat(document.getElementById('symbolic-math-max').value) || 10.0,
            constraints: [document.getElementById('symbolic-math-rule').value]
          };
        } else if (targetType === 'logic') {
          endpoint = '/api/symbolic/verify/logic';
          const pStr = document.getElementById('symbolic-logic-premises').value;
          const premises = pStr.split(',').map(s => s.trim()).filter(Boolean);
          payload = {
            premises: premises,
            hypothesis: document.getElementById('symbolic-logic-hypothesis').value.trim()
          };
        }

        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `HTTP ${res.status}`);
        }
        const result = await res.json();

        if (verdictBox) {
          verdictBox.style.display = 'block';
          if (result.status === 'verified') {
            verdictBox.className = 'alert-box alert-success';
            verdictBox.innerHTML = `<strong>✓ VERIFIED</strong> (${result.verification_latency_ms.toFixed(1)} ms): All formal symbolic constraints satisfied.`;
          } else {
            verdictBox.className = 'alert-box alert-danger';
            verdictBox.innerHTML = `<strong>✗ VIOLATED</strong> (${result.verification_latency_ms.toFixed(1)} ms): ${escapeHtml((result.violations || []).join('; '))}`;
          }
        }
        await loadSymbolicVerifications();
      } catch (err) {
        alert(`Verification failed: ${err.message}`);
      } finally {
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = '⚡ Verify Invariant'; }
      }
    });
  }

  const btnRefreshSymbolic = document.getElementById('btn-refresh-symbolic-table');
  if (btnRefreshSymbolic) btnRefreshSymbolic.addEventListener('click', () => loadSymbolicVerifications());

  // ══════════════════════════════════════════════════════════════
  // ── PHASE 14: AUTONOMOUS RECURSIVE ARCHITECT & MICRO-QUANT ──
  // ══════════════════════════════════════════════════════════════

  async function loadArchitectData() {
    await Promise.all([
      loadPruningExperiments(),
      loadQuantProfiles(),
      loadSpeculativeRuns(),
    ]);
  }

  // 1. Structural Pruning & NAS
  async function loadPruningExperiments() {
    const tbody = document.getElementById('pruning-experiments-tbody');
    if (!tbody) return;
    try {
      const res = await fetch('/api/pruning/experiments');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" style="text-align:center; color:var(--text-muted); padding:20px;">No pruning experiments registered yet. Click \'Plan &amp; Prune\' to begin.</td></tr>';
        return;
      }
      tbody.innerHTML = data.map(exp => {
        const statusBadge = exp.status === 'verified'
          ? '<span class="badge badge-success">✓ Verified</span>'
          : `<span class="badge badge-warning">${escapeHtml(exp.status)}</span>`;
        return `
          <tr>
            <td><code>${escapeHtml(exp.id)}</code></td>
            <td><strong>${escapeHtml(exp.model_name)}</strong></td>
            <td><span class="badge badge-info">${escapeHtml(exp.pruning_strategy)}</span></td>
            <td>${exp.target_vram_mb ? exp.target_vram_mb.toFixed(0) + ' MB' : 'Auto'}</td>
            <td><span style="color:var(--accent-danger);">-${exp.heads_pruned}</span></td>
            <td>${exp.remaining_heads}</td>
            <td><strong>-${exp.parameter_reduction_pct.toFixed(1)}%</strong></td>
            <td>${exp.estimated_vram_mb.toFixed(0)} MB</td>
            <td>${exp.metrics && exp.metrics.speedup_estimate ? exp.metrics.speedup_estimate.toFixed(2) + 'x' : '1.20x'}</td>
            <td>${statusBadge}</td>
            <td style="color:var(--text-muted); font-size:0.75rem;">${escapeHtml(exp.created_at.slice(0, 19).replace('T', ' '))}</td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="11" style="text-align:center; color:var(--accent-danger); padding:20px;">Failed to load pruning experiments: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  // 2. Micro-Quantization Profiles
  async function loadQuantProfiles() {
    const tbody = document.getElementById('quant-profiles-tbody');
    if (!tbody) return;
    try {
      const res = await fetch('/api/quantization/profiles');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" style="text-align:center; color:var(--text-muted); padding:20px;">No micro-quantization profiles benchmarked yet. Click \'Profile Quant\' to test.</td></tr>';
        return;
      }
      tbody.innerHTML = data.map(p => {
        const statusBadge = p.status === 'verified'
          ? '<span class="badge badge-success">✓ Verified</span>'
          : `<span class="badge badge-info">${escapeHtml(p.status)}</span>`;
        return `
          <tr>
            <td><code>${escapeHtml(p.id)}</code></td>
            <td><strong>${escapeHtml(p.model_name)}</strong></td>
            <td><span class="badge badge-warning">${p.bits}-bit</span></td>
            <td>${escapeHtml(p.strategy)}</td>
            <td>Block ${p.block_size}</td>
            <td>${(p.outlier_channel_ratio * 100).toFixed(1)}%</td>
            <td><strong>${p.compression_ratio.toFixed(2)}x</strong></td>
            <td><code>${p.mse_loss.toFixed(6)}</code></td>
            <td><strong style="color:var(--accent-success);">${p.cosine_similarity.toFixed(4)}</strong></td>
            <td>${statusBadge}</td>
            <td style="color:var(--text-muted); font-size:0.75rem;">${escapeHtml(p.created_at.slice(0, 19).replace('T', ' '))}</td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="11" style="text-align:center; color:var(--accent-danger); padding:20px;">Failed to load quantization profiles: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  // 3. Self-Speculative Decoding Runs
  async function loadSpeculativeRuns() {
    const tbody = document.getElementById('speculative-runs-tbody');
    if (!tbody) return;
    try {
      const res = await fetch('/api/speculative/runs');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" style="text-align:center; color:var(--text-muted); padding:20px;">No speculative runs executed yet. Click \'Speculative Run\' to launch.</td></tr>';
        return;
      }
      tbody.innerHTML = data.map(r => {
        const statusBadge = r.status === 'completed'
          ? '<span class="badge badge-success">✓ Accelerated</span>'
          : `<span class="badge badge-warning">${escapeHtml(r.status)}</span>`;
        const preview = r.output_text ? (r.output_text.slice(0, 45) + (r.output_text.length > 45 ? '...' : '')) : '—';
        return `
          <tr>
            <td><code>${escapeHtml(r.id)}</code></td>
            <td>${escapeHtml(r.target_model)}</td>
            <td>${escapeHtml(r.draft_model)}</td>
            <td><span class="badge badge-info">K=${r.lookahead_k}</span></td>
            <td style="font-size:0.8rem; max-width:180px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${escapeHtml(r.output_text || '')}">${escapeHtml(preview)}</td>
            <td>${r.total_draft_tokens}</td>
            <td><strong>${(r.acceptance_rate * 100).toFixed(1)}%</strong></td>
            <td><strong style="color:var(--accent-success); font-size:1.05rem;">${r.speedup_factor.toFixed(2)}x</strong></td>
            <td>${r.speculative_latency_ms.toFixed(0)} ms</td>
            <td>${statusBadge}</td>
            <td style="color:var(--text-muted); font-size:0.75rem;">${escapeHtml(r.created_at.slice(0, 19).replace('T', ' '))}</td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="11" style="text-align:center; color:var(--accent-danger); padding:20px;">Failed to load speculative runs: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  // Modals & Forms wiring
  const pruningModal = document.getElementById('pruning-modal');
  const quantModal = document.getElementById('quant-modal');
  const specModal = document.getElementById('speculative-modal');

  // Pruning modal open/close
  const btnOpenPruning = document.getElementById('btn-open-pruning-modal');
  const btnClosePruning = document.getElementById('btn-close-pruning-modal');
  const btnCancelPruning = document.getElementById('btn-cancel-pruning-modal');
  if (btnOpenPruning && pruningModal) btnOpenPruning.addEventListener('click', () => { pruningModal.style.display = 'flex'; });
  if (btnClosePruning && pruningModal) btnClosePruning.addEventListener('click', () => { pruningModal.style.display = 'none'; });
  if (btnCancelPruning && pruningModal) btnCancelPruning.addEventListener('click', () => { pruningModal.style.display = 'none'; });

  // Quant modal open/close
  const btnOpenQuant = document.getElementById('btn-open-quant-modal');
  const btnCloseQuant = document.getElementById('btn-close-quant-modal');
  const btnCancelQuant = document.getElementById('btn-cancel-quant-modal');
  if (btnOpenQuant && quantModal) btnOpenQuant.addEventListener('click', () => { quantModal.style.display = 'flex'; });
  if (btnCloseQuant && quantModal) btnCloseQuant.addEventListener('click', () => { quantModal.style.display = 'none'; });
  if (btnCancelQuant && quantModal) btnCancelQuant.addEventListener('click', () => { quantModal.style.display = 'none'; });

  // Speculative modal open/close
  const btnOpenSpec = document.getElementById('btn-open-speculative-modal');
  const btnCloseSpec = document.getElementById('btn-close-speculative-modal');
  const btnCancelSpec = document.getElementById('btn-cancel-speculative-modal');
  if (btnOpenSpec && specModal) btnOpenSpec.addEventListener('click', () => { specModal.style.display = 'flex'; });
  if (btnCloseSpec && specModal) btnCloseSpec.addEventListener('click', () => { specModal.style.display = 'none'; });
  if (btnCancelSpec && specModal) btnCancelSpec.addEventListener('click', () => { specModal.style.display = 'none'; });

  // Refresh buttons
  const btnRefPrune = document.getElementById('btn-refresh-pruning-table');
  const btnRefQuant = document.getElementById('btn-refresh-quant-table');
  const btnRefSpec = document.getElementById('btn-refresh-speculative-table');
  if (btnRefPrune) btnRefPrune.addEventListener('click', () => loadPruningExperiments());
  if (btnRefQuant) btnRefQuant.addEventListener('click', () => loadQuantProfiles());
  if (btnRefSpec) btnRefSpec.addEventListener('click', () => loadSpeculativeRuns());

  // Form: Pruning execution
  const pruningForm = document.getElementById('pruning-plan-form');
  if (pruningForm) {
    pruningForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('btn-submit-pruning');
      if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = '⏳ Pruning Architecture...'; }
      try {
        const vramInput = document.getElementById('prune-target-vram');
        const vramVal = vramInput && vramInput.value ? parseFloat(vramInput.value) : null;
        const payload = {
          model_name: document.getElementById('prune-model-select').value,
          strategy: document.getElementById('prune-strategy-select').value,
          target_vram_mb: vramVal,
          head_prune_ratio: parseFloat(document.getElementById('prune-head-ratio').value) || 0.25,
          mlp_prune_ratio: parseFloat(document.getElementById('prune-mlp-ratio').value) || 0.85,
        };
        const res = await fetch('/api/pruning/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        if (pruningModal) pruningModal.style.display = 'none';
        await loadPruningExperiments();
        alert('Structural Pruning experiment completed and verified successfully!');
      } catch (err) {
        alert(`Pruning execution failed: ${err.message}`);
      } finally {
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = '🚀 Plan & Execute Prune'; }
      }
    });
  }

  // Form: Quantization profiling
  const quantForm = document.getElementById('quant-profile-form');
  if (quantForm) {
    quantForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('btn-submit-quant');
      if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = '⏳ Profiling Quantization...'; }
      try {
        const payload = {
          model_name: document.getElementById('quant-model-select').value,
          bits: parseInt(document.getElementById('quant-bits-select').value, 10),
          strategy: document.getElementById('quant-strategy-select').value,
          block_size: parseInt(document.getElementById('quant-block-size').value, 10),
          outlier_channel_ratio: parseFloat(document.getElementById('quant-outlier-ratio').value) || 0.01,
        };
        const res = await fetch('/api/quantization/profile', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        if (quantModal) quantModal.style.display = 'none';
        await loadQuantProfiles();
        alert('Micro-Quantization benchmark profile created and verified successfully!');
      } catch (err) {
        alert(`Quantization profiling failed: ${err.message}`);
      } finally {
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = '⚡ Run Micro-Quant Profile'; }
      }
    });
  }

  // Form: Speculative run
  const specForm = document.getElementById('speculative-run-form');
  if (specForm) {
    specForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('btn-submit-speculative');
      if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = '⏳ Generating Speculatively...'; }
      try {
        const payload = {
          prompt: document.getElementById('spec-prompt-input').value,
          target_model: document.getElementById('spec-target-model').value,
          draft_model: document.getElementById('spec-draft-model').value,
          lookahead_k: parseInt(document.getElementById('spec-lookahead-k').value, 10),
          acceptance_threshold: parseFloat(document.getElementById('spec-threshold').value) || 0.75,
          max_tokens: parseInt(document.getElementById('spec-max-tokens').value, 10) || 64,
        };
        const res = await fetch('/api/speculative/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        if (specModal) specModal.style.display = 'none';
        await loadSpeculativeRuns();
        alert('Speculative generation session completed and verified successfully!');
      } catch (err) {
        alert(`Speculative run failed: ${err.message}`);
      } finally {
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = '🚀 Execute Speculative Run'; }
      }
    });
  }

  // ══════════════════════════════════════════════════════════════
  // ── PHASE 15: WORLD MODELS, LATENT DREAMS & SPECULATIVE MCTS ──
  // ══════════════════════════════════════════════════════════════

  async function loadWorldData() {
    await Promise.all([
      loadWorldTransitions(),
      loadDreamTrajectories(),
      loadMCTSRuns(),
    ]);
  }

  // 1. Multimodal Environment Transitions
  async function loadWorldTransitions() {
    const tbody = document.getElementById('transitions-tbody');
    if (!tbody) return;
    try {
      const res = await fetch('/api/world/transitions');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; color:var(--text-muted); padding:20px;">No transitions recorded yet. Click \'Predict Transition\' to test.</td></tr>';
        return;
      }
      tbody.innerHTML = data.map(t => {
        const errBadge = t.transition_error_score === 0.0
          ? '<span class="badge badge-success">0.000 (Exact)</span>'
          : `<span class="badge badge-warning">${t.transition_error_score.toFixed(3)}</span>`;
        const predBadge = t.safety_predicted
          ? '<span class="badge badge-success">✓ Pass</span>'
          : '<span class="badge badge-danger">✗ Violated</span>';
        const actBadge = t.safety_actual
          ? '<span class="badge badge-success">✓ Pass</span>'
          : '<span class="badge badge-danger">✗ Violated</span>';
        const predFiles = Object.keys(t.predicted_state.files || {}).join(', ') || 'none';
        const actFiles = Object.keys(t.actual_state.files || {}).join(', ') || 'none';
        return `
          <tr>
            <td><code>${escapeHtml(t.id)}</code></td>
            <td><code>${escapeHtml(t.session_id)}</code></td>
            <td><span class="badge badge-info">${escapeHtml(t.action_type)}</span></td>
            <td><small>${escapeHtml(predFiles)}</small></td>
            <td><small>${escapeHtml(actFiles)}</small></td>
            <td>${errBadge}</td>
            <td>${predBadge}</td>
            <td>${actBadge}</td>
            <td style="color:var(--text-muted); font-size:0.75rem;">${escapeHtml((t.created_at || '').slice(0, 19).replace('T', ' '))}</td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:var(--accent-danger); padding:20px;">Failed to load transitions: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  // 2. Latent Dream Trajectories
  async function loadDreamTrajectories() {
    const tbody = document.getElementById('trajectories-tbody');
    if (!tbody) return;
    try {
      const res = await fetch('/api/world/trajectories');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color:var(--text-muted); padding:20px;">No dream trajectories imagined yet. Click \'Simulate Dream\' to explore.</td></tr>';
        return;
      }
      tbody.innerHTML = data.map(t => {
        const valBadge = t.total_value_score >= 0.60
          ? `<strong style="color:var(--accent-success);">${t.total_value_score.toFixed(3)}</strong>`
          : `<span style="color:var(--accent-warning);">${t.total_value_score.toFixed(3)}</span>`;
        const statusBadge = t.status === 'completed'
          ? '<span class="badge badge-success">Completed</span>'
          : `<span class="badge badge-warning">${escapeHtml(t.status)}</span>`;
        return `
          <tr>
            <td><code>${escapeHtml(t.id)}</code></td>
            <td><strong>${escapeHtml(t.prompt)}</strong></td>
            <td>${t.steps ? t.steps.length : t.steps_count}</td>
            <td>${valBadge}</td>
            <td>${t.pruned_branches}</td>
            <td>${statusBadge}</td>
            <td style="color:var(--text-muted); font-size:0.75rem;">${escapeHtml((t.created_at || '').slice(0, 19).replace('T', ' '))}</td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--accent-danger); padding:20px;">Failed to load dream trajectories: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  // 3. Speculative MCTS Runs
  async function loadMCTSRuns() {
    const tbody = document.getElementById('mcts-runs-tbody');
    if (!tbody) return;
    try {
      const res = await fetch('/api/world/mcts-runs');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; color:var(--text-muted); padding:20px;">No MCTS runs launched yet. Click \'Launch MCTS\' to start.</td></tr>';
        return;
      }
      tbody.innerHTML = data.map(r => {
        const statusBadge = r.status === 'completed'
          ? '<span class="badge badge-success">Completed</span>'
          : `<span class="badge badge-warning">${escapeHtml(r.status)}</span>`;
        const pathPreview = (r.best_path || []).slice(0, 2).map(escapeHtml).join(' &rarr; ') || 'Direct Root';
        return `
          <tr>
            <td><code>${escapeHtml(r.id)}</code></td>
            <td><strong>${escapeHtml(r.root_task)}</strong></td>
            <td>${r.iterations}</td>
            <td>${r.nodes_explored}</td>
            <td><strong style="color:var(--accent-success);">${r.best_value.toFixed(3)}</strong></td>
            <td><span style="color:var(--accent-warning);">${r.fork_rollbacks}</span></td>
            <td>${statusBadge}</td>
            <td><small>${pathPreview}</small></td>
            <td style="color:var(--text-muted); font-size:0.75rem;">${escapeHtml((r.created_at || '').slice(0, 19).replace('T', ' '))}</td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:var(--accent-danger); padding:20px;">Failed to load MCTS runs: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  // Modals & Forms
  const transModal = document.getElementById('transition-modal');
  const dreamModal = document.getElementById('dream-modal');
  const mctsModal = document.getElementById('mcts-modal');

  const btnOpenTrans = document.getElementById('btn-open-transition-modal');
  if (btnOpenTrans && transModal) btnOpenTrans.addEventListener('click', () => { transModal.style.display = 'flex'; });
  const btnCloseTrans = document.getElementById('btn-close-transition-modal');
  if (btnCloseTrans && transModal) btnCloseTrans.addEventListener('click', () => { transModal.style.display = 'none'; });
  const btnCancelTrans = document.getElementById('btn-cancel-transition-modal');
  if (btnCancelTrans && transModal) btnCancelTrans.addEventListener('click', () => { transModal.style.display = 'none'; });

  const btnOpenDream = document.getElementById('btn-open-dream-modal');
  if (btnOpenDream && dreamModal) btnOpenDream.addEventListener('click', () => { dreamModal.style.display = 'flex'; });
  const btnCloseDream = document.getElementById('btn-close-dream-modal');
  if (btnCloseDream && dreamModal) btnCloseDream.addEventListener('click', () => { dreamModal.style.display = 'none'; });
  const btnCancelDream = document.getElementById('btn-cancel-dream-modal');
  if (btnCancelDream && dreamModal) btnCancelDream.addEventListener('click', () => { dreamModal.style.display = 'none'; });

  const btnOpenMCTS = document.getElementById('btn-open-mcts-modal');
  if (btnOpenMCTS && mctsModal) btnOpenMCTS.addEventListener('click', () => { mctsModal.style.display = 'flex'; });
  const btnCloseMCTS = document.getElementById('btn-close-mcts-modal');
  if (btnCloseMCTS && mctsModal) btnCloseMCTS.addEventListener('click', () => { mctsModal.style.display = 'none'; });
  const btnCancelMCTS = document.getElementById('btn-cancel-mcts-modal');
  if (btnCancelMCTS && mctsModal) btnCancelMCTS.addEventListener('click', () => { mctsModal.style.display = 'none'; });

  const btnRefreshTrans = document.getElementById('btn-refresh-transitions-table');
  if (btnRefreshTrans) btnRefreshTrans.addEventListener('click', () => loadWorldTransitions());
  const btnRefreshDream = document.getElementById('btn-refresh-trajectories-table');
  if (btnRefreshDream) btnRefreshDream.addEventListener('click', () => loadDreamTrajectories());
  const btnRefreshMCTS = document.getElementById('btn-refresh-mcts-table');
  if (btnRefreshMCTS) btnRefreshMCTS.addEventListener('click', () => loadMCTSRuns());

  // Form: Transition prediction
  const transForm = document.getElementById('transition-predict-form');
  if (transForm) {
    transForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('btn-submit-transition');
      if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = '⏳ Simulating...'; }
      try {
        const payload = {
          current_state: { files: {}, ast_symbols: {} },
          action: {
            action_type: document.getElementById('trans-action-type').value,
            target: document.getElementById('trans-target').value,
            payload: document.getElementById('trans-payload').value,
          },
        };
        const res = await fetch('/api/world/predict-transition', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const pred = await res.json();
        const resultBox = document.getElementById('trans-result-box');
        const badge = document.getElementById('trans-safety-badge');
        const details = document.getElementById('trans-result-details');
        if (resultBox && badge && details) {
          resultBox.style.display = 'block';
          badge.className = pred.safety_score >= 0.8 ? 'badge badge-success' : 'badge badge-danger';
          badge.textContent = `Safety: ${(pred.safety_score * 100).toFixed(0)}%`;
          details.innerHTML = `
            <div><strong>Exit Code:</strong> <code>${pred.predicted_exit_code}</code></div>
            <div><strong>Files Mutated:</strong> ${(pred.delta_files || []).join(', ') || 'None'}</div>
            <div><strong>AST Symbols Added:</strong> ${(pred.delta_ast || []).join('; ') || 'None'}</div>
            ${pred.safety_violations && pred.safety_violations.length > 0 ? `<div style="color:var(--accent-danger); margin-top:4px;"><strong>Violations:</strong> ${pred.safety_violations.map(escapeHtml).join('; ')}</div>` : ''}
          `;
        }

        // Auto-record transition
        await fetch('/api/world/record-transition', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: 'web-session',
            action: payload.action,
            prediction: pred,
            actual_state: pred.predicted_state,
          }),
        });

        await loadWorldTransitions();
      } catch (err) {
        alert(`Transition prediction failed: ${err.message}`);
      } finally {
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = '🔮 Predict Transition'; }
      }
    });
  }

  // Form: Dream rollout
  const dreamForm = document.getElementById('dream-rollout-form');
  if (dreamForm) {
    dreamForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('btn-submit-dream');
      if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = '💭 Imagining Trajectory...'; }
      try {
        const payload = {
          prompt: document.getElementById('dream-prompt-input').value,
          depth: parseInt(document.getElementById('dream-depth').value, 10),
          prune_threshold: parseFloat(document.getElementById('dream-prune-threshold').value) || 0.25,
          export_to_replay: document.getElementById('dream-export-replay').checked,
        };
        const res = await fetch('/api/world/dream-rollout', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const traj = await res.json();
        const view = document.getElementById('dream-trajectory-view');
        if (view) {
          view.style.display = 'block';
          view.innerHTML = (traj.steps || []).map(s => {
            const stepClass = s.pruned ? 'border:1px solid var(--accent-danger); opacity:0.75;' : 'border:1px solid var(--border-color);';
            const badge = s.pruned
              ? '<span class="badge badge-danger">Pruned</span>'
              : `<span class="badge badge-success">V=${s.value_estimate.toFixed(2)}</span>`;
            return `
              <div class="card" style="margin-bottom:8px; padding:8px 12px; background:var(--bg-secondary); ${stepClass}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <strong>Step ${s.step_index}: ${escapeHtml(s.hypothesis_action)}</strong>
                  ${badge}
                </div>
                <div style="font-size:0.8rem; color:var(--text-muted); margin-top:4px;">${escapeHtml(s.imagined_outcome)}</div>
              </div>
            `;
          }).join('');
        }
        await loadDreamTrajectories();
      } catch (err) {
        alert(`Dream simulation failed: ${err.message}`);
      } finally {
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = '💭 Launch Mental Rollout'; }
      }
    });
  }

  // Form: MCTS search
  const mctsForm = document.getElementById('mcts-search-form');
  if (mctsForm) {
    mctsForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('btn-submit-mcts');
      if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = '🌲 Exploring Tree...'; }
      try {
        const payload = {
          root_task: document.getElementById('mcts-task-input').value,
          iterations: parseInt(document.getElementById('mcts-iterations').value, 10),
          max_depth: parseInt(document.getElementById('mcts-depth').value, 10),
        };
        const res = await fetch('/api/world/mcts-search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const run = await res.json();
        const resultBox = document.getElementById('mcts-result-box');
        const badge = document.getElementById('mcts-status-badge');
        const details = document.getElementById('mcts-result-details');
        if (resultBox && badge && details) {
          resultBox.style.display = 'block';
          badge.className = run.status === 'completed' ? 'badge badge-success' : 'badge badge-warning';
          badge.textContent = `${run.status.toUpperCase()} (Value: ${run.best_value.toFixed(2)})`;
          details.innerHTML = `
            <div><strong>Nodes Explored:</strong> ${run.nodes_explored} across ${run.iterations} iterations</div>
            <div><strong>Rollbacks Defended:</strong> ${run.fork_rollbacks}</div>
            <div style="margin-top:6px;"><strong>Best Speculative Trajectory:</strong></div>
            <ol style="margin:4px 0 0 16px; padding:0;">
              ${(run.best_path || []).map(p => `<li>${escapeHtml(p)}</li>`).join('')}
            </ol>
          `;
        }
        await loadMCTSRuns();
      } catch (err) {
        alert(`MCTS search failed: ${err.message}`);
      } finally {
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = '🌲 Start Tree Search'; }
      }
    });
  }

  // ── Initial Bootstrap ──
  loadConversations();
  loadCategories();
  loadHardwareAndModels();
  loadDatasets();
});


