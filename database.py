import sqlite3
import json
from datetime import datetime
import uuid

class ChatDatabase:
    def __init__(self, db_path="chatbot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                theme TEXT DEFAULT 'dark',
                language TEXT DEFAULT 'en',
                notifications BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, username, email=None):
        """Create a new user"""
        user_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (id, username, email)
                VALUES (?, ?, ?)
            ''', (user_id, username, email))
            
            # Create default preferences
            cursor.execute('''
                INSERT INTO user_preferences (user_id)
                VALUES (?)
            ''', (user_id,))
            
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()
    
    def get_user(self, username):
        """Get user by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'created_at': user[3],
                'last_active': user[4]
            }
        return None
    
    def create_session(self, user_id, title="New Chat"):
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (id, user_id, title)
            VALUES (?, ?, ?)
        ''', (session_id, user_id, title))
        
        conn.commit()
        conn.close()
        return session_id
    
    def get_user_sessions(self, user_id):
        """Get all sessions for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, title, created_at, updated_at
            FROM sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
        ''', (user_id,))
        
        sessions = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': session[0],
                'title': session[1],
                'created_at': session[2],
                'updated_at': session[3]
            }
            for session in sessions
        ]
    
    def add_message(self, session_id, role, content, confidence=None):
        """Add a message to a session"""
        message_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages (id, session_id, role, content, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (message_id, session_id, role, content, confidence))
        
        # Update session timestamp
        cursor.execute('''
            UPDATE sessions
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (session_id,))
        
        conn.commit()
        conn.close()
        return message_id
    
    def get_session_messages(self, session_id):
        """Get all messages for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT role, content, confidence, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
        ''', (session_id,))
        
        messages = cursor.fetchall()
        conn.close()
        
        return [
            {
                'role': message[0],
                'content': message[1],
                'confidence': message[2],
                'created_at': message[3]
            }
            for message in messages
        ]
    
    def update_session_title(self, session_id, title):
        """Update session title"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE sessions
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (title, session_id))
        
        conn.commit()
        conn.close()
    
    def delete_session(self, session_id):
        """Delete a session and all its messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
        
        conn.commit()
        conn.close()
    
    def get_user_preferences(self, user_id):
        """Get user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_preferences WHERE user_id = ?', (user_id,))
        prefs = cursor.fetchone()
        conn.close()
        
        if prefs:
            return {
                'theme': prefs[1],
                'language': prefs[2],
                'notifications': bool(prefs[3])
            }
        return None
    
    def update_user_preferences(self, user_id, **preferences):
        """Update user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for key, value in preferences.items():
            cursor.execute(f'''
                UPDATE user_preferences
                SET {key} = ?
                WHERE user_id = ?
            ''', (value, user_id))
        
        conn.commit()
        conn.close()
