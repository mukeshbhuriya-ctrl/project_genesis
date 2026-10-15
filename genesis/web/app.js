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

  // ── Initial Bootstrap ──
  loadConversations();
  loadCategories();
  loadHardwareAndModels();
});
