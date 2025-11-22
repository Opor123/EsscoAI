// script.js - Enhanced security version
const sideNav = document.querySelector('.sideNav');
const input = document.getElementById('chat-input');
const chat = document.getElementById('chatbox');
const API_BASE = window.__ESSCO_API_BASE__ || '/api';

let isLoading = false;
let conversations = [];
let adminToken = null; // Store admin token in memory only

const mainEl = document.querySelector('main');
const featureSection = document.getElementById('feature');
const adminSection = document.getElementById('admin-dashboard');
const adminLink = document.getElementById('adminLink');
const adminLoginModal = document.getElementById('adminLoginModal');

// ===========================
// SECURE SESSION MANAGEMENT
// ===========================

function getAdminToken() {
    return adminToken;
}

function setAdminToken(token) {
    adminToken = token;
    // Do NOT store in localStorage or sessionStorage for security
}

function clearAdminToken() {
    adminToken = null;
}

function isAdminLoggedIn() {
    return adminToken !== null;
}

// ===========================
// SECURE API CALLS
// ===========================

async function secureApiCall(endpoint, options = {}) {
    const headers = {
        'Content-Type': 'application/json',
        'X-Session-Id': currentSessionId,
        ...options.headers
    };

    // Add admin token if available
    if (adminToken) {
        headers['Authorization'] = `Bearer ${adminToken}`;
    }

    const response = await fetch(`${API_BASE}${endpoint}`, {
        ...options,
        headers
    });

    // Check for auth errors
    if (response.status === 401) {
        clearAdminToken();
        updateAdminLinkVisibility();
        if (endpoint.startsWith('/admin')) {
            showToast('Session expired. Please login again.', 'error');
            showAdminLogin();
        }
        throw new Error('Unauthorized');
    }

    return response;
}

// ===========================
// ADMIN LOGIN
// ===========================

function showAdminLogin() {
    if (adminLoginModal) {
        adminLoginModal.style.display = 'flex';
        document.getElementById('adminLoginUser').value = '';
        document.getElementById('adminLoginPass').value = '';
        document.getElementById('loginError').textContent = '';
        setTimeout(() => document.getElementById('adminLoginUser').focus(), 50);
    }
}

function closeAdminLogin() {
    if (adminLoginModal) {
        adminLoginModal.style.display = 'none';
        document.getElementById('loginError').textContent = '';
        document.getElementById('adminLoginUser').value = '';
        document.getElementById('adminLoginPass').value = '';
    }
}

async function attemptAdminLogin() {
    const username = document.getElementById('adminLoginUser').value.trim();
    const password = document.getElementById('adminLoginPass').value;
    const errorEl = document.getElementById('loginError');

    if (!username || !password) {
        errorEl.textContent = 'Please enter both username and password';
        return;
    }

    errorEl.textContent = 'Verifying...';

    try {
        const res = await fetch(`${API_BASE}/admin/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Session-Id': currentSessionId
            },
            body: JSON.stringify({ username, password })
        });

        const data = await res.json();

        if (res.ok && data.success) {
            // Store token securely in memory
            setAdminToken(data.token);

            closeAdminLogin();
            updateAdminLinkVisibility();
            showToast(`✅ Logged in as ${data.username}`, 'success');

            // Auto-open admin dashboard
            setTimeout(() => {
                if (adminLink) adminLink.click();
            }, 300);

        } else {
            errorEl.textContent = '❌ ' + (data.message || 'Invalid credentials');
            document.getElementById('adminLoginPass').value = '';
            document.getElementById('adminLoginPass').focus();
        }
    } catch (e) {
        errorEl.textContent = '❌ Connection error. Please try again.';
        console.error('Admin login error:', e);
    }
}

async function adminLogout() {
    if (!confirm('Are you sure you want to logout?')) return;

    try {
        await secureApiCall('/admin/logout', { method: 'POST' });
    } catch (e) {
        console.error('Logout error:', e);
    }

    clearAdminToken();
    updateAdminLinkVisibility();
    showToast('🔒 Logged out', 'info');

    // Return to main chat
    hideFeature();
    if (adminSection) adminSection.style.display = 'none';
    if (mainEl) mainEl.style.display = 'flex';
}

function updateAdminLinkVisibility() {
    if (adminLink) {
        adminLink.style.display = isAdminLoggedIn() ? 'flex' : 'none';
    }
}

// ===========================
// ADMIN DASHBOARD
// ===========================

async function loadAdminDashboard() {
    if (!isAdminLoggedIn()) {
        showAdminLogin();
        return;
    }

    try {
        const res = await secureApiCall('/admin/dashboard', {
            method: 'GET',
            cache: 'no-store'
        });

        if (!res.ok) {
            throw new Error('Failed to load dashboard');
        }

        const d = await res.json();

        document.getElementById('adm-total-conv').textContent = d.total_conversations;
        document.getElementById('adm-total-msg').textContent = d.total_messages;
        document.getElementById('adm-last24-msg').textContent = d.last_24h_messages;

        document.getElementById('adm-qa-loaded').textContent = d.qa_system_loaded ? 'Yes' : 'No';
        document.getElementById('adm-qa-pairs').textContent = d.total_qa_pairs ?? '-';

        document.getElementById('adm-llm-avail').textContent = d.llm_available ? 'Yes' : 'No';
        document.getElementById('adm-llm-enabled').textContent = d.llm_enabled ? 'Yes' : 'No';
        document.getElementById('adm-llm-model').textContent = d.llm_model || '-';

        document.getElementById('adm-fb-total').textContent = d.feedback_total;
        document.getElementById('adm-fb-up').textContent = d.feedback_up;
        document.getElementById('adm-fb-down').textContent = d.feedback_down;
        document.getElementById('adm-fb-neutral').textContent = d.feedback_neutral;

        addLogoutButton();
    } catch (e) {
        console.error('Admin dashboard error:', e);
        showToast('Failed to load admin dashboard', 'error');
        if (e.message === 'Unauthorized') {
            showAdminLogin();
        }
    }
}

function addLogoutButton() {
    if (adminSection && !document.getElementById('adminLogoutBtn')) {
        const logoutBtn = document.createElement('button');
        logoutBtn.id = 'adminLogoutBtn';
        logoutBtn.className = 'admin-logout-btn';
        logoutBtn.innerHTML = '<i class="fi fi-rr-sign-out-alt"></i> Logout';
        logoutBtn.onclick = adminLogout;
        adminSection.appendChild(logoutBtn);
    }
}

// ===========================
// RETRAIN (ADMIN ONLY)
// ===========================

const retrainBtn = document.getElementById('btnRetrain');
const retrainMsg = document.getElementById('retrainMsg');

if (retrainBtn) {
    // Remove old password fields (no longer needed)
    const adminUser = document.getElementById('adminUser');
    const adminPass = document.getElementById('adminPass');
    const saveCredsBtn = document.getElementById('saveAdminCreds');

    if (adminUser) adminUser.style.display = 'none';
    if (adminPass) adminPass.style.display = 'none';
    if (saveCredsBtn) saveCredsBtn.style.display = 'none';

    retrainBtn.addEventListener('click', async () => {
        if (!isAdminLoggedIn()) {
            showToast('Please login as admin first', 'error');
            showAdminLogin();
            return;
        }

        retrainMsg.textContent = 'Retraining...';
        retrainBtn.disabled = true;

        try {
            const res = await secureApiCall('/retrain', {
                method: 'POST'
            });

            const data = await res.json().catch(() => ({}));

            if (!res.ok) throw new Error(data?.detail || res.statusText);

            retrainMsg.textContent = data?.message || 'Retrained successfully';
            showToast('✅ Model retrained', 'success');
        } catch (e) {
            retrainMsg.textContent = `Failed: ${e.message}`;
            showToast('Failed to retrain model', 'error');
        } finally {
            retrainBtn.disabled = false;
        }
    });
}

// ===========================
// CHAT FUNCTIONALITY
// ===========================

function generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

let currentSessionId = localStorage.getItem("ESSCO_SESSION_ID") || generateSessionId();
localStorage.setItem("ESSCO_SESSION_ID", currentSessionId);

function showFunction() {
    if (!featureSection) return;
    featureSection.style.display = 'block';
    if (mainEl) mainEl.style.visibility = 'hidden';
    if (adminSection) adminSection.style.display = 'none';
    refreshStatus();
}

function hideFeature() {
    if (!featureSection) return;
    featureSection.style.display = 'none';
    if (mainEl) mainEl.style.visibility = 'visible';
    if (adminSection) adminSection.style.display = 'none';
}

async function init() {
    if (localStorage.getItem("ESSCO_SIDENAV_COLLAPSED") === '1') {
        document.body.classList.add('sideNav-collapse');
    }

    await loadConversations();
    setupEventListeners();

    try {
        await refreshStatus();
    } catch (e) {
        console.warn('Status refresh failed on init:', e);
    }

    updateAdminLinkVisibility();
    input.focus();
}

document.getElementById('chat-form').addEventListener('submit', (e) => {
    e.preventDefault();
    if (!isLoading) sendMsg();
});

function setupEventListeners() {
    document.querySelector('.close-open').addEventListener('click', (e) => {
        e.preventDefault();
        document.body.classList.toggle('sideNav-collapse');
        localStorage.setItem('ESSCO_SIDENAV_COLLAPSED',
            document.body.classList.contains('sideNav-collapse') ? '1' : '0'
        );
    });

    const newChatLink = document.querySelector('.el .nav-link[href="#"]');
    if (newChatLink) {
        newChatLink.addEventListener('click', (e) => {
            e.preventDefault();
            hideFeature();
            if (adminSection) adminSection.style.display = 'none';
            startNewChat();
        });
    }

    const featureLink = document.querySelector('.nav-link[href="#feature"]');
    if (featureLink && featureSection) {
        featureLink.addEventListener('click', (e) => {
            e.preventDefault();
            const visible = getComputedStyle(featureSection).display !== 'none';
            if (visible) {
                hideFeature();
            } else {
                showFunction();
            }
        });
    }

    if (adminLink && adminSection) {
        adminLink.addEventListener('click', (e) => {
            e.preventDefault();

            if (!isAdminLoggedIn()) {
                showAdminLogin();
                return;
            }

            if (featureSection) featureSection.style.display = "none";
            if (mainEl) mainEl.style.display = "none";
            adminSection.style.display = "block";

            document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));
            adminLink.classList.add('active');

            loadAdminDashboard();
        });
    }

    // Login modal events
    if (adminLoginModal) {
        adminLoginModal.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') attemptAdminLogin();
        });

        adminLoginModal.addEventListener('click', (e) => {
            if (e.target === adminLoginModal) closeAdminLogin();
        });
    }
}

async function loadConversations() {
    hideFeature();
    try {
        const res = await secureApiCall('/conversations?limit=50', {
            cache: "no-store"
        });
        if (!res.ok) throw new Error('Failed to load conversations');

        conversations = await res.json();
        renderConversations();
    } catch (e) {
        console.error('Error loading conversations: ', e);
    }
}

function renderConversations() {
    const container = document.querySelector('.sideNav');
    const oldList = document.getElementById('conversation-list');
    if (oldList) oldList.remove();

    const listDiv = document.createElement('div');
    listDiv.id = 'conversation-list';
    listDiv.className = 'conversation-list';

    if (conversations.length === 0) {
        listDiv.innerHTML = '<div class="no-conversations">No conversations yet</div>';
    } else {
        conversations.forEach(con => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            if (con.session_id === currentSessionId) {
                item.classList.add('active');
            }

            item.innerHTML = `
            <div class="conversation-content" data-session="${con.session_id}">
                <div class="conversation-title">${escapeHtml(con.title)}</div>
                <div class="conversation-preview">${escapeHtml(con.preview)}</div>
                <div class="conversation-meta">${formatTime(con.last_message_time)} • ${con.message_count} msgs</div>
            </div>
            <button class="delete-conversation" data-session="${con.session_id}" title="Delete">
                <i class="fi fi-rr-trash"></i>
            </button>
            `;

            item.querySelector('.conversation-content').addEventListener('click', () => {
                loadConversation(con.session_id);
            });

            item.querySelector('.delete-conversation').addEventListener('click', (e) => {
                e.stopPropagation();
                deleteConversation(con.session_id);
            });

            listDiv.appendChild(item);
        });
    }

    const bottomLine = container.querySelector('.bottom-line');
    bottomLine.after(listDiv);

    ensureConversationSearch();
}

function ensureConversationSearch() {
    if (document.querySelector('.conversation-search')) return;
    addConversationSearch();
}

async function loadConversation(sessionId) {
    hideFeature();
    try {
        const res = await secureApiCall(`/conversations/${sessionId}?limit=200&offset=0`);
        if (!res.ok) throw new Error('Failed to load conversation');

        const data = await res.json();
        chat.innerHTML = '';

        const welcome = document.getElementById('welcome');
        if (welcome) welcome.remove();

        currentSessionId = sessionId;
        localStorage.setItem("ESSCO_SESSION_ID", currentSessionId);

        data.messages.forEach(msg => {
            if (msg.user_message) {
                addMsg(msg.user_message, 'user', false, msg.timestamp);
            }
            if (msg.bot_answer) {
                addMsg(msg.bot_answer, 'bot', false, msg.timestamp);
            }
        });

        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.remove('active');
            if (item.querySelector(`[data-session="${sessionId}"]`)) {
                item.classList.add('active');
            }
        });

        input.focus();
    } catch (e) {
        console.error('Error loading conversation:', e);
        showToast('Failed to load conversation', 'error');
    }
}

async function deleteConversation(sessionId) {
    if (!confirm('Delete this conversation?')) return;
    try {
        const res = await secureApiCall(`/conversations/${sessionId}`, {
            method: 'DELETE'
        });

        if (!res.ok) throw new Error('Failed to delete conversation');

        if (sessionId === currentSessionId) {
            startNewChat();
        }
        await loadConversations();
    } catch (error) {
        console.error('Error deleting conversation:', error);
        showToast('Failed to delete conversation', 'error');
    }
}

function startNewChat() {
    currentSessionId = generateSessionId();
    localStorage.setItem("ESSCO_SESSION_ID", currentSessionId);
    chat.innerHTML = '';

    if (!document.getElementById('welcome')) {
        const welcome = document.createElement('h1');
        welcome.id = 'welcome';
        welcome.textContent = 'Welcome to Essco';
        document.querySelector('main').insertBefore(welcome, chat);
    }

    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.remove('active');
    });

    if (adminSection) adminSection.style.display = 'none';
    if (featureSection) featureSection.style.display = 'none';
    if (mainEl) mainEl.style.display = 'flex';

    input.value = '';
    input.focus();
    loadConversations().catch(() => {});
}

const submitBtn = document.getElementById('submitMsg');

async function sendMsg() {
    const text = input.value.trim();
    if (!text || isLoading) return;

    isLoading = true;
    submitBtn && (submitBtn.disabled = true);

    addMsg(text, 'user');
    input.value = '';

    const loadingId = addTypingIndicator();

    try {
        const res = await secureApiCall('/chat-llm', {
            method: 'POST',
            body: JSON.stringify({
                message: text,
                session_id: currentSessionId,
            }),
        });

        if (res.status === 429) {
            removeMsg(loadingId);
            addMsg("Rate limit hit. Please try again in a minute.", 'bot');
            return;
        }

        if (!res.ok) {
            if (res.status === 503) {
                removeMsg(loadingId);
                addMsg("Service is warming up or temporarily unavailable. Please try again shortly.", "bot");
                return;
            }
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }

        const data = await res.json();
        const ans = data.answer || data.message || "No response received";

        const welcome = document.getElementById('welcome');
        if (welcome) {
            welcome.style.animation = "fadeOut 0.3s ease";
            setTimeout(() => welcome.remove(), 300);
        }

        removeMsg(loadingId);
        const mid = addMsg(ans, 'bot');

        if (data.meta?.llm_active) {
            const llmBadge = document.createElement("div");
            llmBadge.className = 'llm-badge';
            llmBadge.innerHTML = '<i class="fi fi-rr-sparkles"></i> AI Enhanced';
            llmBadge.title = data.meta?.reason || 'Enhanced with AI for better response';
            document.getElementById(mid)?.appendChild(llmBadge);
        }

        if (typeof (data.conf) === 'number') {
            const badge = document.createElement("div");
            const lvl = data.conf >= 0.7 ? 'high' : (data.conf >= 0.4 ? 'medium' : 'low');
            badge.className = `confidence-badge confidence-${lvl}`;
            badge.textContent = `${Math.round(data.conf * 100)}% ${data?.meta?.confidence_label || 'confidence'}`;
            document.getElementById(mid)?.appendChild(badge);
        }

        await loadConversations();
    } catch (error) {
        console.error('API Error:', error);
        removeMsg(loadingId);
        addMsg(`Error: Couldn't connect to API - ${error.message}`, 'bot');
    } finally {
        isLoading = false;
        submitBtn && (submitBtn.disabled = false);
        input.focus();
    }
}

async function refreshStatus() {
    const el = document.getElementById('statusText');
    try {
        const res = await secureApiCall('/status', {
            cache: 'no-store'
        });
        let data = {};
        try {
            data = await res.json();
        } catch {
        }
        if (el) {
            el.textContent = `Loaded: ${data.qa_system_loaded ?? 'n/a'} | QA pairs: ${data.total_qa_pairs ?? 'n/a'}`;
        }

        // Update admin link visibility based on login status OR config
        if (adminLink) {
            if (isAdminLoggedIn()) {
                adminLink.style.display = 'flex';
            } else if (data.admin_link_enabled) {
                adminLink.style.display = 'flex';
            } else {
                // Keep hidden unless logged in or explicitly enabled
                adminLink.style.display = 'none';
            }
        }
    } catch {
        if (el) el.textContent = 'Failed to load status';
    }
}

document.getElementById('refreshStatus')?.addEventListener('click', refreshStatus);

function addMsg(text, sender, animate = true, tsSec = null) {
    const div = document.createElement('div');
    div.classList.add('msg', sender);
    const content = document.createElement('div');
    content.className = 'msg-content';
    content.textContent = text;

    const stamp = document.createElement('div');
    stamp.className = 'msg-timestamp';
    const when = tsSec ? new Date(tsSec * 1000) : new Date();
    stamp.textContent = when.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});

    div.appendChild(content);
    div.appendChild(stamp);

    const messageId = `msg-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    div.id = messageId;

    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;

    return messageId;
}

function removeMsg(messageId) {
    const ele = document.getElementById(messageId);
    if (ele) {
        ele.style.animation = 'fadeOut 0.2s ease';
        setTimeout(() => ele.remove(), 200);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(timestamp) {
    const date = new Date(timestamp * 1000);
    const now = new Date();
    const diff = now - date;

    if (diff < 60000) return 'Just Now';
    if (diff < 3600000) {
        const mins = Math.floor(diff / 60000);
        return `${mins}m ago`;
    }
    if (diff < 86400000) {
        const hours = Math.floor(diff / 3600000);
        return `${hours}h ago`;
    }
    if (diff < 604800000) {
        const days = Math.floor(diff / 86400000);
        return `${days}d ago`;
    }
    return date.toLocaleDateString();
}

let fbLabel = 'neutral';
document.getElementById('fbUp')?.addEventListener('click', () => fbLabel = 'up');
document.getElementById('fbDown')?.addEventListener('click', () => fbLabel = 'down');
document.getElementById('fbSend')?.addEventListener('click', async () => {
    const note = document.getElementById('fbNote').value.trim();
    const payload = {label: fbLabel, user_comment: note || undefined, session_id: currentSessionId};
    const msg = document.getElementById('fbMsg');
    try {
        const res = await secureApiCall('/feedback', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
        if (res.status === 429) {
            msg.textContent = 'Too many requests. Try later.';
            return;
        }
        if (!res.ok) throw new Error(await res.text());
        msg.textContent = 'Thanks! Feedback recorded.';
        document.getElementById('fbNote').value = '';
        fbLabel = 'neutral';
    } catch (e) {
        msg.textContent = 'Failed to submit feedback';
    }
});

function addConversationSearch() {
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = 'Search conversations...';
    searchInput.className = 'conversation-search';

    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        document.querySelectorAll('.conversation-item').forEach(item => {
            const title = item.querySelector('.conversation-title').textContent.toLowerCase();
            const preview = item.querySelector('.conversation-preview').textContent.toLowerCase();
            item.style.display = (title.includes(query) || preview.includes(query)) ? 'flex' : 'none';
        });
    });

    const firstEl = document.querySelector('.sideNav .el');
    firstEl.parentNode.insertBefore(searchInput, firstEl);
}

function addTypingIndicator() {
    const wrap = document.createElement('div');
    wrap.className = 'msg bot typing-indicator';
    wrap.innerHTML = `
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
    `;
    const id = `msg-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    wrap.id = id;
    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
    return id;
}

function showToast(message, type = 'info') {
    const t = document.createElement('div');
    t.className = `toast toast-${type}`;
    t.textContent = message;
    document.body.appendChild(t);
    requestAnimationFrame(() => t.classList.add('show'));
    setTimeout(() => {
        t.classList.remove('show');
        setTimeout(() => t.remove(), 300);
    }, 3000);
}

const style = document.createElement('style');
style.textContent = `
@keyframes fadeOut {
    from { opacity: 1; transform: scale(1); }
    to { opacity: 0; transform: scale(0.95); }
}
`;
document.head.appendChild(style);

// Keyboard shortcut to open admin (Ctrl+Shift+A)
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.shiftKey && e.key === 'A') {
        e.preventDefault();
        if (!isAdminLoggedIn()) {
            showAdminLogin();
        } else {
            // Open admin dashboard
            if (adminLink) adminLink.click();
        }
    }
});

// Allow direct admin access via URL hash
function checkAdminHash() {
    if (window.location.hash === '#admin') {
        if (!isAdminLoggedIn()) {
            showAdminLogin();
        } else {
            if (adminLink) adminLink.click();
        }
        // Clear hash after handling
        history.replaceState(null, null, ' ');
    }
}

window.addEventListener('hashchange', checkAdminHash);

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        init();
        checkAdminHash();
    });
} else {
    init();
    checkAdminHash();
}