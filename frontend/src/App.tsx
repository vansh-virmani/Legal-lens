import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, Send, FileText, Activity, Clock, ShieldAlert } from 'lucide-react';
import './App.css';

type MessageType = 'user' | 'rag' | 'risk' | 'summary' | 'timeline';

interface Message {
  id: string;
  sender: 'user' | 'ai';
  type: MessageType;
  content: any; // Can be string, or structured object/array based on type
}

function App() {
  const [docId, setDocId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isSending]);

  const handleFileUpload = async (file: File) => {
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.doc_id) {
        setDocId(data.doc_id);
        setMessages([
          {
            id: Date.now().toString(),
            sender: 'ai',
            type: 'rag',
            content: `Document successfully uploaded! What would you like to know about it? You can ask for a summary, risk analysis, timeline, or specific questions.`
          }
        ]);
      } else {
        alert(data.error || 'Upload failed');
      }
    } catch (err) {
      console.error(err);
      alert('Error connecting to backend.');
    } finally {
      setIsUploading(false);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
      handleFileUpload(file);
    } else {
      alert('Please upload a PDF file.');
    }
  };

  const handleSend = async () => {
    if (!input.trim() || !docId) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      sender: 'user',
      type: 'user',
      content: input
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsSending(true);

    try {
      const res = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id: docId, query: userMsg.content }),
      });
      
      const data = await res.json();
      
      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
        type: data.type || 'rag',
        content: data.type === 'rag' ? data.answer : data.result
      };
      
      setMessages(prev => [...prev, aiMsg]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        sender: 'ai',
        type: 'rag',
        content: 'Sorry, there was an error processing your request.'
      }]);
    } finally {
      setIsSending(false);
    }
  };

  const renderMessageContent = (msg: Message) => {
    if (msg.sender === 'user') {
      return <p style={{ margin: 0 }}>{msg.content}</p>;
    }

    switch (msg.type) {
      case 'summary':
        return (
          <div className="summary-block">
            <h4 style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: 0, color: 'var(--primary)' }}>
              <FileText size={18} /> Summary
            </h4>
            <p style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{msg.content}</p>
          </div>
        );
      
      case 'risk':
        const riskContent = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
        
        // Simple heuristic to color-code risk levels
        const renderRiskText = (text: string) => {
          const lines = text.split('\n').filter(l => l.trim() !== '');
          return lines.map((line, idx) => {
            let badgeClass = '';
            let badgeText = '';
            const lowerLine = line.toLowerCase();
            if (lowerLine.includes('high risk') || lowerLine.match(/\bhigh\b/)) {
              badgeClass = 'high'; badgeText = 'High';
            } else if (lowerLine.includes('medium risk') || lowerLine.match(/\bmedium\b/)) {
              badgeClass = 'medium'; badgeText = 'Medium';
            } else if (lowerLine.includes('low risk') || lowerLine.match(/\blow\b/)) {
              badgeClass = 'low'; badgeText = 'Low';
            }
            
            return (
              <div key={idx} className="risk-item">
                {badgeClass && <span className={`risk-badge ${badgeClass}`}>{badgeText}</span>}
                <div style={{ flex: 1 }}>{line}</div>
              </div>
            );
          });
        };

        return (
          <div className="risk-list">
            <h4 style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: 0, color: 'var(--danger)' }}>
              <ShieldAlert size={18} /> Risk Analysis
            </h4>
            {renderRiskText(riskContent)}
          </div>
        );

      case 'timeline':
        return (
          <div>
            <h4 style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: 0, color: 'var(--warning)' }}>
              <Clock size={18} /> Timeline Events
            </h4>
            <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
          </div>
        );

      case 'rag':
      default:
        return <p style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{msg.content}</p>;
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Legal Lens</h1>
        <p>AI-Powered Document Intelligence</p>
      </header>

      {!docId ? (
        <div 
          className="glass-panel upload-zone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <UploadCloud className="upload-icon" size={64} />
          <h2 style={{ margin: '0 0 0.5rem 0' }}>Upload Legal Document</h2>
          <p style={{ color: 'var(--text-muted)', margin: 0 }}>Drag & drop your PDF here, or click to browse</p>
          <input 
            type="file" 
            accept=".pdf" 
            ref={fileInputRef} 
            style={{ display: 'none' }} 
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleFileUpload(e.target.files[0]);
              }
            }}
          />
          {isUploading && (
            <div style={{ marginTop: '2rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--primary)' }}>
              <Activity size={20} className="upload-icon" /> Processing Document...
            </div>
          )}
        </div>
      ) : (
        <div className="glass-panel chat-container">
          <div className="chat-history">
            {messages.map((msg) => (
              <div key={msg.id} className={`message ${msg.sender}`}>
                {renderMessageContent(msg)}
              </div>
            ))}
            {isSending && (
              <div className="message ai">
                <div className="loading-dots">
                  <div></div><div></div><div></div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-area">
            <input 
              type="text" 
              placeholder="Ask about risks, summary, dates, or specific clauses..." 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSend();
              }}
              disabled={isSending}
            />
            <button onClick={handleSend} disabled={!input.trim() || isSending}>
              <Send size={20} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
