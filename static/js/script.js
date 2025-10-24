// Enhanced Neural Network Chatbot JavaScript
class NeuralNetworkChatbot {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.queryInput = document.getElementById('query');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.themeToggle = document.getElementById('themeToggle');
        this.voiceBtn = document.getElementById('voiceBtn');
        this.exportBtn = document.getElementById('exportBtn');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.sidebarToggle = document.getElementById('sidebarToggle');
        this.sidebar = document.getElementById('sidebar');
        this.sessionsList = document.getElementById('sessionsList');
        this.userInfo = document.getElementById('userInfo');
        this.username = document.getElementById('username');
        this.logoutBtn = document.getElementById('logoutBtn');
        
        // Modals
        this.loginModal = document.getElementById('loginModal');
        this.fileUploadModal = document.getElementById('fileUploadModal');
        
        // Forms
        this.loginForm = document.getElementById('loginForm');
        this.registerForm = document.getElementById('registerForm');
        
        // State
        this.isProcessing = false;
        this.messageCount = 0;
        this.isDarkMode = true;
        this.currentSessionId = null;
        this.isLoggedIn = false;
        this.recognition = null;
        this.isRecording = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.autoResizeTextarea();
        this.checkServerHealth();
        this.initializeTheme();
        this.initializeVoiceRecognition();
        this.checkAuthStatus();
        this.loadSessions();
    }
    
    setupEventListeners() {
        // Chat functionality
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        this.queryInput.addEventListener('input', () => {
            this.autoResizeTextarea();
            this.toggleSendButton();
        });
        
        // Theme toggle
        this.themeToggle.addEventListener('click', () => this.toggleTheme());
        
        // Voice recognition
        this.voiceBtn.addEventListener('click', () => this.toggleVoiceRecording());
        
        // Export functionality
        this.exportBtn.addEventListener('click', () => this.exportChat());
        
        // New chat
        this.newChatBtn.addEventListener('click', () => this.createNewChat());
        
        // Sidebar toggle
        this.sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        
        // Logout
        this.logoutBtn.addEventListener('click', () => this.logout());
        
        // Modal functionality
        this.setupModalListeners();
        
        // Auth forms
        this.loginForm.addEventListener('submit', (e) => this.handleLogin(e));
        this.registerForm.addEventListener('submit', (e) => this.handleRegister(e));
        
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
        
        // File upload
        this.setupFileUpload();
    }
    
    setupModalListeners() {
        // Close modals when clicking outside
        window.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                e.target.style.display = 'none';
            }
        });
        
        // Close buttons
        document.querySelectorAll('.close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.target.closest('.modal').style.display = 'none';
            });
        });
    }
    
    setupFileUpload() {
        const fileUploadArea = document.getElementById('fileUploadArea');
        const fileInput = document.getElementById('fileInput');
        
        fileUploadArea.addEventListener('click', () => fileInput.click());
        fileUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUploadArea.style.borderColor = 'var(--primary-color)';
        });
        
        fileUploadArea.addEventListener('dragleave', () => {
            fileUploadArea.style.borderColor = 'var(--border-color)';
        });
        
        fileUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUploadArea.style.borderColor = 'var(--border-color)';
            const files = e.dataTransfer.files;
            this.handleFileUpload(files);
        });
        
        fileInput.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });
    }
    
    async checkAuthStatus() {
        try {
            const response = await fetch('/api/sessions');
            if (response.ok) {
                this.isLoggedIn = true;
                this.showUserInterface();
                this.loadSessions();
            } else {
                this.showLoginModal();
            }
        } catch (error) {
            console.log('Not authenticated, showing login modal');
            this.showLoginModal();
        }
    }
    
    showLoginModal() {
        this.loginModal.style.display = 'block';
    }
    
    showUserInterface() {
        this.userInfo.style.display = 'flex';
        this.loadSessions();
    }
    
    async handleLogin(e) {
        e.preventDefault();
        const username = document.getElementById('loginUsername').value;
        const password = document.getElementById('loginPassword').value;
        
        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.isLoggedIn = true;
                this.username.textContent = username;
                this.loginModal.style.display = 'none';
                this.showUserInterface();
                this.loadSessions();
            } else {
                const error = await response.json();
                alert(error.error || 'Login failed');
            }
        } catch (error) {
            console.error('Login error:', error);
            alert('Login failed. Please try again.');
        }
    }
    
    async handleRegister(e) {
        e.preventDefault();
        const username = document.getElementById('registerUsername').value;
        const password = document.getElementById('registerPassword').value;
        
        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.isLoggedIn = true;
                this.username.textContent = username;
                this.loginModal.style.display = 'none';
                this.showUserInterface();
                this.loadSessions();
            } else {
                const error = await response.json();
                alert(error.error || 'Registration failed');
            }
        } catch (error) {
            console.error('Registration error:', error);
            alert('Registration failed. Please try again.');
        }
    }
    
    switchTab(tabName) {
        // Switch auth tabs
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        document.getElementById(`${tabName}Tab`).classList.add('active');
    }
    
    async loadSessions() {
        if (!this.isLoggedIn) return;
        
        try {
            const response = await fetch('/api/sessions');
            if (response.ok) {
                const data = await response.json();
                this.renderSessions(data.sessions);
            }
        } catch (error) {
            console.error('Failed to load sessions:', error);
        }
    }
    
    renderSessions(sessions) {
        this.sessionsList.innerHTML = '';
        
        sessions.forEach(session => {
            const sessionElement = document.createElement('div');
            sessionElement.className = 'session-item';
            sessionElement.dataset.sessionId = session.id;
            
            sessionElement.innerHTML = `
                <div class="session-title">${session.title}</div>
                <div class="session-date">${new Date(session.updated_at).toLocaleDateString()}</div>
                <div class="session-actions">
                    <button class="session-action" onclick="chatbot.deleteSession('${session.id}')" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            `;
            
            sessionElement.addEventListener('click', () => {
                this.loadSession(session.id);
            });
            
            this.sessionsList.appendChild(sessionElement);
        });
    }
    
    async loadSession(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}/messages`);
            if (response.ok) {
                const data = await response.json();
                this.currentSessionId = sessionId;
                this.clearChat();
                this.renderMessages(data.messages);
                this.updateActiveSession(sessionId);
            }
        } catch (error) {
            console.error('Failed to load session:', error);
        }
    }
    
    renderMessages(messages) {
        messages.forEach(message => {
            this.addMessage(message.content, message.role, message.confidence);
        });
    }
    
    updateActiveSession(sessionId) {
        document.querySelectorAll('.session-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-session-id="${sessionId}"]`).classList.add('active');
    }
    
    async createNewChat() {
        if (!this.isLoggedIn) {
            this.showLoginModal();
        return;
    }

        try {
            const response = await fetch('/api/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: 'New Chat' })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.currentSessionId = data.session_id;
                this.clearChat();
                this.loadSessions();
            }
        } catch (error) {
            console.error('Failed to create new chat:', error);
        }
    }
    
    async deleteSession(sessionId) {
        if (!confirm('Are you sure you want to delete this chat?')) return;
        
        try {
            const response = await fetch(`/api/sessions/${sessionId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                this.loadSessions();
                if (this.currentSessionId === sessionId) {
                    this.currentSessionId = null;
                    this.clearChat();
                }
            }
        } catch (error) {
            console.error('Failed to delete session:', error);
        }
    }
    
    async logout() {
        try {
            await fetch('/api/logout', { method: 'POST' });
            this.isLoggedIn = false;
            this.userInfo.style.display = 'none';
            this.currentSessionId = null;
            this.clearChat();
            this.showLoginModal();
        } catch (error) {
            console.error('Logout failed:', error);
        }
    }
    
    clearChat() {
        this.chatMessages.innerHTML = `
            <div class="welcome-message">
                <div class="bot-message">
                    <div class="message-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-content">
                        <p>Hello! I'm your Neural Network Assistant. I can help you with questions about:</p>
                        <ul>
                            <li>Neural Networks (CNN, RNN, LSTM, GAN, etc.)</li>
                            <li>Machine Learning concepts</li>
                            <li>Deep Learning implementations</li>
                            <li>Code examples and best practices</li>
                        </ul>
                        <p>What would you like to know?</p>
                    </div>
                </div>
            </div>
        `;
    }
    
    toggleSidebar() {
        this.sidebar.classList.toggle('collapsed');
    }
    
    initializeVoiceRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-US';
            
            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                this.queryInput.value = transcript;
                this.autoResizeTextarea();
                this.toggleSendButton();
            };
            
            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.isRecording = false;
                this.voiceBtn.classList.remove('active');
            };
            
            this.recognition.onend = () => {
                this.isRecording = false;
                this.voiceBtn.classList.remove('active');
            };
        }
    }
    
    toggleVoiceRecording() {
        if (!this.recognition) {
            alert('Voice recognition is not supported in your browser.');
            return;
        }
        
        if (this.isRecording) {
            this.recognition.stop();
        } else {
            this.recognition.start();
            this.isRecording = true;
            this.voiceBtn.classList.add('active');
        }
    }
    
    exportChat() {
        const messages = this.chatMessages.querySelectorAll('.message');
        let exportText = 'Neural Network Assistant Chat Export\n';
        exportText += '=====================================\n\n';
        
        messages.forEach(message => {
            const role = message.classList.contains('user-message') ? 'User' : 'Assistant';
            const content = message.querySelector('.message-content').textContent;
            exportText += `${role}: ${content}\n\n`;
        });
        
        const blob = new Blob([exportText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat-export-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    handleFileUpload(files) {
        Array.from(files).forEach(file => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const content = e.target.result;
                this.queryInput.value = `Please analyze this ${file.name} file:\n\n${content}`;
                this.autoResizeTextarea();
                this.toggleSendButton();
            };
            reader.readAsText(file);
        });
        this.fileUploadModal.style.display = 'none';
    }
    
    autoResizeTextarea() {
        this.queryInput.style.height = 'auto';
        this.queryInput.style.height = Math.min(this.queryInput.scrollHeight, 120) + 'px';
    }
    
    toggleSendButton() {
        const hasText = this.queryInput.value.trim().length > 0;
        this.sendButton.disabled = !hasText || this.isProcessing;
    }
    
    async checkServerHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            this.updateStatus('healthy', `Ready (${data.dataset_size} Q&As loaded)`);
        } catch (error) {
            this.updateStatus('error', 'Server connection failed');
        }
    }
    
    updateStatus(status, message) {
        const icon = this.statusIndicator.querySelector('i');
        const text = this.statusIndicator.querySelector('span');
        
        icon.className = 'fas fa-circle';
        text.textContent = message;
        
        switch (status) {
            case 'healthy':
                icon.style.color = '#10b981';
                break;
            case 'processing':
                icon.style.color = '#f59e0b';
                break;
            case 'error':
                icon.style.color = '#ef4444';
                break;
        }
    }
    
    async sendMessage() {
        const query = this.queryInput.value.trim();
        if (!query || this.isProcessing) return;
        
        this.addMessage(query, 'user');
        this.queryInput.value = '';
        this.autoResizeTextarea();
        this.toggleSendButton();
        
        this.showTypingIndicator();
        this.updateStatus('processing', 'AI is thinking...');
        this.isProcessing = true;
        
        try {
            const response = await fetch('/ask', {
                method: 'POST',
        headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    query,
                    session_id: this.currentSessionId
                }),
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            
        if (data.error) {
                throw new Error(data.error);
            }
            
            this.addBotMessage(data.answer, data.accuracy);
            this.updateStatus('healthy', 'Ready');
            
            // Update current session ID if we got one
            if (data.session_id) {
                this.currentSessionId = data.session_id;
                this.loadSessions();
            }
            
        } catch (error) {
            console.error('Error:', error);
            this.addErrorMessage(error.message);
            this.updateStatus('error', 'Error occurred');
        } finally {
            this.hideTypingIndicator();
            this.isProcessing = false;
            this.toggleSendButton();
        }
    }
    
    addMessage(content, type, confidence = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        if (type === 'user') {
            messageContent.innerHTML = `<p>${this.escapeHtml(content)}</p>`;
        } else {
            messageContent.innerHTML = this.formatBotMessage(content);
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        // Add reactions for bot messages
        if (type === 'assistant') {
            const reactions = document.createElement('div');
            reactions.className = 'message-reactions';
            reactions.innerHTML = `
                <button class="reaction-btn" onclick="chatbot.addReaction(this, '👍')">👍</button>
                <button class="reaction-btn" onclick="chatbot.addReaction(this, '👎')">👎</button>
                <button class="reaction-btn" onclick="chatbot.addReaction(this, '❤️')">❤️</button>
            `;
            messageDiv.appendChild(reactions);
        }
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    addBotMessage(content, accuracy) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '<i class="fas fa-robot"></i>';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = this.formatBotMessage(content);
        
        // Add accuracy badge
        const accuracyBadge = document.createElement('div');
        accuracyBadge.className = `accuracy-badge ${this.getAccuracyClass(accuracy)}`;
        accuracyBadge.textContent = `Confidence: ${(accuracy * 100).toFixed(1)}%`;
        messageContent.appendChild(accuracyBadge);
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        // Add reactions
        const reactions = document.createElement('div');
        reactions.className = 'message-reactions';
        reactions.innerHTML = `
            <button class="reaction-btn" onclick="chatbot.addReaction(this, '👍')">👍</button>
            <button class="reaction-btn" onclick="chatbot.addReaction(this, '👎')">👎</button>
            <button class="reaction-btn" onclick="chatbot.addReaction(this, '❤️')">❤️</button>
        `;
        messageDiv.appendChild(reactions);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    addErrorMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content error-message';
        messageContent.innerHTML = `<p>Sorry, I encountered an error: ${this.escapeHtml(message)}</p>`;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    addReaction(button, emoji) {
        button.classList.toggle('active');
        // Here you could save reactions to the database
    }
    
    formatBotMessage(content) {
        // Format code blocks
        let formatted = this.escapeHtml(content);
        
        // Convert code blocks with syntax highlighting
        formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            return `
                <div class="code-block-container">
                    <div class="code-header">
                        <span class="code-language">${lang || 'code'}</span>
                        <button class="copy-button" onclick="chatbot.copyCode(this)">
                            <i class="fas fa-copy"></i> Copy
                        </button>
                    </div>
                    <pre class="code-block"><code class="language-${lang || 'python'}">${code.trim()}</code></pre>
                </div>
            `;
        });
        
        // Convert inline code
        formatted = formatted.replace(/`([^`]+)`/g, '<code style="background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-family: monospace;">$1</code>');
        
        // Convert line breaks
        formatted = formatted.replace(/\n/g, '<br>');
        
        // Convert lists
        formatted = formatted.replace(/^[\s]*[-*+]\s+(.+)$/gm, '<li>$1</li>');
        formatted = formatted.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        
        return `<p>${formatted}</p>`;
    }
    
    copyCode(button) {
        const codeBlock = button.closest('.code-block-container').querySelector('.code-block');
        const text = codeBlock.textContent;
        
        navigator.clipboard.writeText(text).then(() => {
            button.innerHTML = '<i class="fas fa-check"></i> Copied!';
            setTimeout(() => {
                button.innerHTML = '<i class="fas fa-copy"></i> Copy';
            }, 2000);
        });
    }
    
    getAccuracyClass(accuracy) {
        if (accuracy >= 0.7) return '';
        if (accuracy >= 0.4) return 'low';
        return 'very-low';
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    showTypingIndicator() {
        this.typingIndicator.classList.add('show');
    }
    
    hideTypingIndicator() {
        this.typingIndicator.classList.remove('show');
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    initializeTheme() {
        // Load theme preference from localStorage
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            this.isDarkMode = false;
            document.body.classList.add('light-mode');
        }
        
        // Update theme toggle icon
        const icon = this.themeToggle.querySelector('i');
        icon.className = this.isDarkMode ? 'fas fa-moon' : 'fas fa-lightbulb';
    }
    
    toggleTheme() {
        this.isDarkMode = !this.isDarkMode;
        document.body.classList.toggle('light-mode', !this.isDarkMode);
        
        // Update theme toggle icon
        const icon = this.themeToggle.querySelector('i');
        icon.className = this.isDarkMode ? 'fas fa-moon' : 'fas fa-lightbulb';
        
        // Save preference to localStorage
        localStorage.setItem('theme', this.isDarkMode ? 'dark' : 'light');
    }
}

// Initialize the chatbot when the DOM is loaded
let chatbot;
document.addEventListener('DOMContentLoaded', () => {
    chatbot = new NeuralNetworkChatbot();
});

// Add some helpful keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K to focus input
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        document.getElementById('query').focus();
    }
    
    // Escape to clear input
    if (e.key === 'Escape') {
        document.getElementById('query').value = '';
        document.getElementById('query').style.height = 'auto';
        document.getElementById('sendButton').disabled = true;
    }
    
    // Ctrl/Cmd + N for new chat
    if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
        e.preventDefault();
        chatbot.createNewChat();
    }
});