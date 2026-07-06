'use client';

import { useState, useEffect, useRef, useCallback, Suspense } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { useSearchParams, useRouter } from 'next/navigation';
import { signInWithGoogle } from '@/lib/firebase';

// ── Lightweight markdown → HTML ──────────────────────────────────
function parseMarkdown(raw) {
  let t = raw
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  t = t.replace(/```[\w]*\n?([\s\S]*?)```/g, (_, c) =>
    `<pre><code>${c.trim()}</code></pre>`
  );
  t = t.replace(/^---$/gm, '<hr/>');
  t = t.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  t = t.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  t = t.replace(/^# (.+)$/gm, '<h2>$1</h2>');
  t = t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/\*(.+?)\*/g, '<em>$1</em>');
  t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
  t = t.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
  t = t.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');

  t = t.replace(/((?:^\|.+\|\n?)+)/gm, (block) => {
    const rows = block.trim().split('\n');
    const isSep = (r) => /^[\|\s\-:]+$/.test(r);
    let html = '<table>';
    rows.forEach((row, i) => {
      if (isSep(row)) return;
      const isHeader = i === 0 && rows.length > 1 && isSep(rows[1]);
      const tag = isHeader ? 'th' : 'td';
      const cells = row.replace(/^\|/, '').replace(/\|$/, '').split('|');
      html += '<tr>' + cells.map((c) => `<${tag}>${c.trim()}</${tag}>`).join('') + '</tr>';
    });
    return html + '</table>';
  });

  const lines = t.split('\n');
  const result = [];
  let inPre = false;
  for (const line of lines) {
    if (line.startsWith('<pre')) inPre = true;
    if (line.startsWith('</pre')) inPre = false;
    if (
      inPre ||
      line.startsWith('<h') ||
      line.startsWith('<ul') ||
      line.startsWith('<li') ||
      line.startsWith('<table') ||
      line.startsWith('<tr') ||
      line.startsWith('<hr') ||
      line.trim() === ''
    ) {
      result.push(line);
    } else {
      result.push(`<p>${line}</p>`);
    }
  }
  return result.join('\n');
}

// ── Icon paths ───────────────────────────────────────────────────
const D = {
  send:    'M2.01 21L23 12 2.01 3 2 10l15 2-15 2z',
  trash:   'M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z',
  export:  'M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z',
  copy:    'M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z',
  check:   'M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z',
  plus:    'M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z',
  menu:    'M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z',
  close:   'M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z',
  logout:  'M17 7l-1.41 1.41L18.17 11H8v2h10.17l-2.58 2.58L17 17l5-5zM4 5h8V3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8v-2H4V5z',
  chat:    'M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z',
  eye:     'M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z',
  eyeOff:  'M12 7c2.76 0 5 2.24 5 5 0 .65-.13 1.26-.36 1.83l2.92 2.92c1.51-1.26 2.7-2.89 3.43-4.75-1.73-4.39-6-7.5-11-7.5-1.4 0-2.74.25-3.98.7l2.16 2.16C10.74 7.13 11.35 7 12 7zM2 4.27l2.28 2.28.46.46C3.08 8.3 1.78 10.02 1 12c1.73 4.39 6 7.5 11 7.5 1.55 0 3.03-.3 4.38-.84l.42.42L19.73 22 21 20.73 3.27 3 2 4.27zM7.53 9.8l1.55 1.55c-.05.21-.08.43-.08.65 0 1.66 1.34 3 3 3 .22 0 .44-.03.65-.08l1.55 1.55c-.67.33-1.41.53-2.2.53-2.76 0-5-2.24-5-5 0-.79.2-1.53.53-2.2zm4.31-.78l3.15 3.15.02-.16c0-1.66-1.34-3-3-3l-.17.01z',
};

const Ico = ({ d, size = 15 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d={d} />
  </svg>
);

// ── Inline SVG brand mark ────────────────────────────────────────
const ArohaLogo = ({ size = 24 }) => (
  <Image src="/aroha-logo.jpeg" alt="Aroha" width={size} height={size} className="rounded-lg object-cover" />
);

// ── Markdown renderer ────────────────────────────────────────────
function MD({ text }) {
  return (
    <div className="md" style={{ fontSize: 14 }}
      dangerouslySetInnerHTML={{ __html: parseMarkdown(text) }} />
  );
}

// ── Typing dots ──────────────────────────────────────────────────
function Dots() {
  return (
    <div className="flex gap-[5px] py-[2px] items-center">
      {[0, 1, 2].map((i) => (
        <span key={i} className="inline-block w-[6px] h-[6px] rounded-full bg-[#4457F5] opacity-50"
          style={{ animation: `chatBounce 1.1s ${i * 0.16}s infinite` }} />
      ))}
    </div>
  );
}

// ── Copy button ──────────────────────────────────────────────────
function CopyBtn({ text }) {
  const [ok, setOk] = useState(false);
  return (
    <button onClick={() => {
      navigator.clipboard.writeText(text).then(() => {
        setOk(true); setTimeout(() => setOk(false), 2000);
      });
    }} className="flex items-center gap-1 px-[7px] py-[3px] rounded text-[11px] transition-colors duration-150 cursor-pointer"
      style={{ color: ok ? '#10B981' : '#6B7280' }}>
      <Ico d={ok ? D.check : D.copy} size={11} />
      {ok ? 'Copied' : 'Copy'}
    </button>
  );
}

// ── Message bubble ───────────────────────────────────────────────
function Message({ msg }) {
  const isUser = msg.role === 'user';
  const isErr  = msg.isError;
  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}
      style={{ animation: 'chatFadeUp .25s cubic-bezier(.22,1,.36,1) both' }}>
      <div className="text-[14px] leading-[1.7] break-words max-w-[78%]"
        style={{
          padding: isUser ? '11px 15px' : '0',
          borderRadius: isUser ? '10px 2px 10px 10px' : '0',
          ...(isUser
            ? { background: 'linear-gradient(135deg,#5B6CF9,#4457F5)', color: '#fff', boxShadow: '0 2px 12px rgba(68,87,245,0.18)' }
            : isErr
            ? { background: 'rgba(220,38,38,.05)', border: '1px solid rgba(220,38,38,.2)', color: '#DC2626', padding: '11px 15px', borderRadius: '10px' }
            : { background: 'transparent', color: '#1F2937' }),
        }}>
        {isUser ? <span>{msg.text}</span> : <MD text={msg.text} />}
        {!isUser && !isErr && <div className="pt-1"><CopyBtn text={msg.text} /></div>}
      </div>
    </div>
  );
}

// ── Starter chips ────────────────────────────────────────────────
const CHIPS = [
  'Dengue outbreaks in Tamil Nadu',
  'Cholera cases in Maharashtra',
  'Malaria deaths in Odisha',
  'Outbreaks in Delhi',
  'Recent Kerala disease alerts',
  'Top diseases by case count',
];

function Welcome({ onChip }) {
  const [hover, setHover] = useState(null);
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-6 py-10"
      style={{ animation: 'chatFadeIn .4s ease' }}>
      <h1 className="text-[28px] font-extrabold text-[#111827] mb-[6px]" style={{ letterSpacing:'-.02em' }}>Aroha</h1>
      <div className="flex items-center gap-2 mb-3">
        <div className="h-px w-10" style={{ background:'#E8002D' }} />
        <span className="text-[11px] font-semibold uppercase" style={{ color:'#E8002D', letterSpacing:'.12em' }}>India Idsp surveillance</span>
        <div className="h-px w-10" style={{ background:'#E8002D' }} />
      </div>
      <p className="text-[14px] text-center leading-[1.75] mb-9 max-w-[380px]" style={{ color:'#6B7280' }}>
        Ask about disease outbreaks, case counts, death tolls, and trends across India's surveillance dataset.
      </p>
      <div className="flex flex-wrap gap-2 justify-center max-w-[540px]">
        {CHIPS.map((c, i) => (
          <button key={i} onClick={() => onChip(c)}
            onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
            className="text-[12px] font-medium rounded-[6px] cursor-pointer transition-all duration-150"
            style={{
              padding: '7px 15px', letterSpacing: '.01em',
              background: hover===i ? 'rgba(68,87,245,0.06)' : '#F9FAFB',
              border: hover===i ? '1px solid rgba(68,87,245,.35)' : '1px solid #E5E7EB',
              color: hover===i ? '#4457F5' : '#6B7280',
            }}>
            {c}
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Input bar ────────────────────────────────────────────────────
function InputBar({ onSend, disabled }) {
  const [text, setText] = useState('');
  const [focus, setFocus] = useState(false);
  const ref = useRef(null);

  useEffect(() => { if (!disabled) ref.current?.focus(); }, [disabled]);

  const resize = () => {
    const el = ref.current; if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 150) + 'px';
  };
  const submit = () => {
    const t = text.trim(); if (!t || disabled) return;
    onSend(t); setText('');
    if (ref.current) ref.current.style.height = 'auto';
  };

  const today = new Date().toLocaleDateString('en-IN', { day:'numeric', month:'short', year:'numeric' }).toUpperCase();

  return (
    <div className="flex-shrink-0 relative z-[2]" style={{ padding:'10px 18px 16px', background: '#FFFFFF', borderTop: '1px solid #F3F4F6' }}>
      <div style={{ maxWidth:760, margin:'0 auto' }}>
        <div style={{
          background:'#FFFFFF',
          border:`1px solid ${focus?'rgba(68,87,245,.45)':'#E5E7EB'}`,
          boxShadow: focus?'0 0 0 3px rgba(68,87,245,.07)':'0 1px 3px rgba(0,0,0,0.04)',
          transition:'border-color .2s,box-shadow .2s',
          overflow:'hidden', borderRadius:10,
        }}>
          <div className="flex items-end gap-2" style={{ padding:'10px 10px 10px 14px' }}>
            <textarea ref={ref} value={text} rows={1} disabled={disabled}
              placeholder="Ask about outbreaks, states, diseases…"
              onChange={(e) => { setText(e.target.value); resize(); }}
              onKeyDown={(e) => { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submit();} }}
              onFocus={() => setFocus(true)} onBlur={() => setFocus(false)}
              className="flex-1 bg-transparent border-none outline-none resize-none text-[14px] leading-[1.6] text-[#111827] placeholder-[#D1D5DB] font-sans"
              style={{ maxHeight:150, padding:'2px 0', caretColor:'#4457F5' }} />
            <button onClick={submit} disabled={!text.trim()||disabled}
              className="flex-shrink-0 w-[34px] h-[34px] rounded-[6px] flex items-center justify-center transition-all duration-150"
              style={{
                background: (text.trim()&&!disabled)?'linear-gradient(135deg,#5B6CF9,#4457F5)':'#F3F4F6',
                border: 'none',
                cursor: (text.trim()&&!disabled)?'pointer':'not-allowed',
                color: (text.trim()&&!disabled)?'#fff':'#D1D5DB',
                boxShadow: (text.trim()&&!disabled)?'0 2px 8px rgba(68,87,245,0.22)':'none',
              }}>
              <Ico d={D.send} size={14} />
            </button>
          </div>
          <div className="flex items-center justify-between" style={{ padding:'5px 12px 8px', borderTop:'1px solid #F3F4F6' }}>
            <span className="text-[10px] font-medium" style={{ color:'#9CA3AF', letterSpacing:'.04em' }}>India Idsp · {today}</span>
            <span className="text-[10px]" style={{ color:'#9CA3AF' }}>
              <kbd style={{ background:'#F9FAFB', border:'1px solid #E5E7EB', borderRadius:3, padding:'1px 5px', fontSize:9, color:'#9CA3AF', fontFamily:'inherit' }}>↵</kbd>
              {' '}send
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Auth Modal (Login / Signup) ───────────────────────────────────
function AuthModal({ defaultTab = 'login', onClose, onAuth }) {
  const [tab, setTab] = useState(defaultTab);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError(''); setLoading(true);
    try {
      const endpoint = tab === 'login' ? '/api/auth/login' : '/api/auth/signup';
      const body = tab === 'login' ? { email, password } : { name, email, password };
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.error || 'Something went wrong.'); return; }
      onAuth(data.user);
    } catch { setError('Network error. Please try again.'); }
    finally { setLoading(false); }
  };

  const handleGoogleSignIn = async () => {
    setError('');
    setLoading(true);
    try {
      const { idToken } = await signInWithGoogle();
      const res = await fetch('/api/auth/google', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ idToken }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || 'Google sign-in failed.');
        return;
      }
      onAuth(data.user);
    } catch (err) {
      console.error('[Google Auth Error]', err);
      setError('Google sign-in failed. Make sure you completed the Firebase configuration steps.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background:'rgba(17,24,39,0.35)', backdropFilter:'blur(4px)' }}
      onClick={(e) => { if(e.target===e.currentTarget) onClose(); }}>
      <div className="relative w-full max-w-[400px] mx-4 rounded-2xl overflow-hidden"
        style={{ background:'#FFFFFF', border:'1px solid #E5E7EB', boxShadow:'0 20px 60px rgba(0,0,0,0.12)', animation:'chatFadeUp .3s ease' }}>

        {/* Red top stripe */}
        <div className="h-[2px] w-full" style={{ background:'#E8002D' }} />

        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-5 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center">
              <ArohaLogo size={22} />
            </div>
            <span className="font-bold text-[#111827] text-[15px]">Aroha</span>
          </div>
          <button onClick={onClose} className="w-7 h-7 flex items-center justify-center rounded-lg transition-colors" style={{ color:'#9CA3AF' }}
            onMouseEnter={(e) => e.currentTarget.style.color='#374151'} onMouseLeave={(e) => e.currentTarget.style.color='#9CA3AF'}>
            <Ico d={D.close} size={16} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex mx-6 mb-5 rounded-lg overflow-hidden" style={{ background:'#F9FAFB', border:'1px solid #E5E7EB' }}>
          {['login','signup'].map((t) => (
            <button key={t} onClick={() => { setTab(t); setError(''); }}
              className="flex-1 py-2 text-[13px] font-semibold transition-all duration-200"
              style={{
                background: tab===t ? 'linear-gradient(135deg,#5B6CF9,#4457F5)' : 'transparent',
                color: tab===t ? '#fff' : '#6B7280',
              }}>
              {t === 'login' ? 'Sign in' : 'Sign up'}
            </button>
          ))}
        </div>

        {/* Form */}
        <form onSubmit={submit} className="px-6 pb-6 flex flex-col gap-3">
          {tab === 'signup' && (
            <div className="flex flex-col gap-1.5">
              <label className="text-[11px] font-semibold uppercase" style={{ color:'#6B7280', letterSpacing:'.08em' }}>Full name</label>
              <input value={name} onChange={(e) => setName(e.target.value)} required placeholder="Your name"
                className="w-full px-3 py-2.5 rounded-lg text-[14px] outline-none transition-all text-[#111827]"
                style={{ background:'#F9FAFB', border:'1px solid #E5E7EB', color:'#111827' }}
                onFocus={(e) => { e.target.style.borderColor='rgba(68,87,245,.5)'; e.target.style.background='#fff'; }}
                onBlur={(e) => { e.target.style.borderColor='#E5E7EB'; e.target.style.background='#F9FAFB'; }} />
            </div>
          )}

          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-semibold uppercase" style={{ color:'#6B7280', letterSpacing:'.08em' }}>Email</label>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required placeholder="you@example.com"
              className="w-full px-3 py-2.5 rounded-lg text-[14px] outline-none transition-all"
              style={{ background:'#F9FAFB', border:'1px solid #E5E7EB', color:'#111827' }}
              onFocus={(e) => { e.target.style.borderColor='rgba(68,87,245,.5)'; e.target.style.background='#fff'; }}
              onBlur={(e) => { e.target.style.borderColor='#E5E7EB'; e.target.style.background='#F9FAFB'; }} />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-semibold uppercase" style={{ color:'#6B7280', letterSpacing:'.08em' }}>Password</label>
            <div className="relative">
              <input type={showPw?'text':'password'} value={password} onChange={(e) => setPassword(e.target.value)} required
                placeholder={tab==='signup' ? 'Min. 6 characters' : 'Your password'}
                className="w-full px-3 py-2.5 pr-10 rounded-lg text-[14px] outline-none transition-all"
                style={{ background:'#F9FAFB', border:'1px solid #E5E7EB', color:'#111827' }}
                onFocus={(e) => { e.target.style.borderColor='rgba(68,87,245,.5)'; e.target.style.background='#fff'; }}
                onBlur={(e) => { e.target.style.borderColor='#E5E7EB'; e.target.style.background='#F9FAFB'; }} />
              <button type="button" onClick={() => setShowPw(!showPw)}
                className="absolute right-3 top-1/2 -translate-y-1/2" style={{ color:'#9CA3AF' }}>
                <Ico d={showPw ? D.eyeOff : D.eye} size={16} />
              </button>
            </div>
          </div>

          {error && (
            <div className="px-3 py-2 rounded-lg text-[13px]"
              style={{ background:'rgba(220,38,38,.05)', border:'1px solid rgba(220,38,38,.2)', color:'#DC2626' }}>
              {error}
            </div>
          )}

          <button type="submit" disabled={loading}
            className="w-full py-3 mt-1 rounded-lg text-[14px] font-semibold text-white transition-all"
            style={{
              background: loading?'#E5E7EB':'linear-gradient(135deg,#5B6CF9,#4457F5)',
              cursor: loading?'not-allowed':'pointer',
              color: loading?'#9CA3AF':'#fff',
              boxShadow: loading?'none':'0 3px 14px rgba(68,87,245,0.28)',
            }}>
            {loading ? 'Please wait…' : tab==='login' ? 'Sign in' : 'Create account'}
          </button>

          {/* Divider */}
          <div className="flex items-center my-1.5">
            <div className="flex-1 h-px" style={{ background:'#E5E7EB' }} />
            <span className="px-3 text-[10px] font-bold text-[#9CA3AF] uppercase tracking-wider">or</span>
            <div className="flex-1 h-px" style={{ background:'#E5E7EB' }} />
          </div>

          {/* Google Button */}
          <button type="button" disabled={loading} onClick={handleGoogleSignIn}
            className="w-full py-3 rounded-lg text-[14px] font-semibold transition-all flex items-center justify-center gap-2"
            style={{
              background: '#F9FAFB',
              border: '1px solid #E5E7EB',
              color: '#374151',
              cursor: loading?'not-allowed':'pointer',
            }}
            onMouseEnter={(e) => { if(!loading) { e.currentTarget.style.background='#F3F4F6'; e.currentTarget.style.borderColor='#D1D5DB'; } }}
            onMouseLeave={(e) => { if(!loading) { e.currentTarget.style.background='#F9FAFB'; e.currentTarget.style.borderColor='#E5E7EB'; } }}>
            <svg className="w-4 h-4 mr-0.5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" fill="#FBBC05" />
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" fill="#EA4335" />
            </svg>
            Continue with Google
          </button>

          <p className="text-center text-[12px] mt-1" style={{ color:'#9CA3AF' }}>
            {tab==='login' ? "Don't have an account? " : 'Already have an account? '}
            <button type="button" onClick={() => { setTab(tab==='login'?'signup':'login'); setError(''); }}
              style={{ color:'#4457F5' }} className="hover:underline">
              {tab==='login' ? 'Sign up' : 'Sign in'}
            </button>
          </p>
        </form>
      </div>
    </div>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────
function Sidebar({ open, user, conversations, activeConvId, onNewChat, onSelectConv, onDeleteConv, onShowAuth, onLogout, onToggleSidebar }) {
  const [hovConv, setHovConv] = useState(null);
  const [showUserMenu, setShowUserMenu] = useState(false);

  const grouped = (() => {
    const today = new Date(); today.setHours(0,0,0,0);
    const yesterday = new Date(today); yesterday.setDate(today.getDate()-1);
    const last7 = new Date(today); last7.setDate(today.getDate()-7);

    const groups = { Today:[], Yesterday:[], 'Last 7 days':[], Older:[] };
    conversations.forEach((c) => {
      const d = new Date(c.updatedAt); d.setHours(0,0,0,0);
      if (d >= today) groups['Today'].push(c);
      else if (d >= yesterday) groups['Yesterday'].push(c);
      else if (d >= last7) groups['Last 7 days'].push(c);
      else groups['Older'].push(c);
    });
    return groups;
  })();

  return (
    <>
      {open && <div className="fixed inset-0 z-20 lg:hidden" style={{ background:'rgba(0,0,0,0.25)' }} onClick={onToggleSidebar} />}

      <aside
        className="flex-shrink-0 flex flex-col z-30 transition-all duration-300 overflow-hidden"
        style={{
          width: open ? 260 : 0,
          background: '#FAFAFA',
          borderRight: open ? '1px solid #E5E7EB' : 'none',
          minWidth: open ? 260 : 0,
        }}>
        {open && (
          <div className="flex flex-col h-full overflow-hidden" style={{ minWidth:260 }}>
            {/* Sidebar header */}
            <div className="flex-shrink-0 px-3 pt-4 pb-3" style={{ borderBottom:'1px solid #E5E7EB' }}>
              <div className="flex items-center gap-2 px-2 mb-3">
                <div className="w-6 h-6 rounded flex-shrink-0 flex items-center justify-center">
                  <ArohaLogo size={20} />
                </div>
                <span className="font-bold text-[#111827] text-[13px]" style={{ letterSpacing:'.03em' }}>Aroha</span>
                <span className="text-[8px] font-bold px-1.5 py-0.5 rounded" style={{ color:'#E8002D', background:'rgba(232,0,45,.07)', border:'1px solid rgba(232,0,45,.18)', letterSpacing:'.08em' }}>Idsp</span>
              </div>

              {user && (
                <button onClick={onNewChat}
                  className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-[13px] font-medium transition-all duration-150"
                  style={{ background:'rgba(68,87,245,0.06)', border:'1px solid rgba(68,87,245,0.2)', color:'#4457F5' }}
                  onMouseEnter={(e) => { e.currentTarget.style.background='rgba(68,87,245,0.1)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background='rgba(68,87,245,0.06)'; }}>
                  <Ico d={D.plus} size={14} /> New chat
                </button>
              )}
            </div>

            {/* Conversation list OR guest CTA */}
            <div className="flex-1 overflow-y-auto py-2 chat-scroll">
              {!user ? (
                <div className="px-4 py-6 flex flex-col items-center text-center gap-4">
                  <div className="w-12 h-12 rounded-xl flex items-center justify-center" style={{ background:'rgba(68,87,245,0.07)', border:'1px solid rgba(68,87,245,0.15)' }}>
                    <Ico d={D.chat} size={22} />
                  </div>
                  <div>
                    <p className="text-[13px] font-semibold text-[#111827] mb-1">Save your chat history</p>
                    <p className="text-[12px] leading-[1.6]" style={{ color:'#6B7280' }}>
                      Sign in to save conversations, access history, and sync across devices.
                    </p>
                  </div>
                  <button onClick={() => onShowAuth('login')}
                    className="w-full py-2 rounded-lg text-[13px] font-semibold text-white transition-all"
                    style={{ background:'linear-gradient(135deg,#5B6CF9,#4457F5)', boxShadow:'0 3px 12px rgba(68,87,245,0.25)' }}>
                    Sign in
                  </button>
                  <button onClick={() => onShowAuth('signup')}
                    className="w-full py-2 rounded-lg text-[13px] font-medium transition-all text-[#374151]"
                    style={{ background:'transparent', border:'1px solid #E5E7EB' }}
                    onMouseEnter={(e) => { e.currentTarget.style.borderColor='#D1D5DB'; e.currentTarget.style.background='#F9FAFB'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.borderColor='#E5E7EB'; e.currentTarget.style.background='transparent'; }}>
                    Create account
                  </button>
                  <p className="text-[11px]" style={{ color:'#9CA3AF' }}>
                    You can still chat without signing in — history won't be saved.
                  </p>
                </div>
              ) : conversations.length === 0 ? (
                <div className="px-4 py-6 text-center">
                  <p className="text-[12px]" style={{ color:'#9CA3AF' }}>No conversations yet.<br />Start a new chat above!</p>
                </div>
              ) : (
                Object.entries(grouped).map(([group, convs]) =>
                  convs.length === 0 ? null : (
                    <div key={group} className="mb-2">
                      <p className="px-4 py-1 text-[10px] font-semibold uppercase" style={{ color:'#9CA3AF', letterSpacing:'.08em' }}>{group}</p>
                      {convs.map((conv) => (
                        <div key={conv._id}
                          className="group relative flex items-center mx-2 mb-0.5 rounded-lg cursor-pointer transition-all duration-150"
                          style={{
                            background: activeConvId===conv._id ? 'rgba(68,87,245,0.08)' : 'transparent',
                            border: activeConvId===conv._id ? '1px solid rgba(68,87,245,0.2)' : '1px solid transparent',
                          }}
                          onMouseEnter={(e) => { setHovConv(conv._id); if(activeConvId!==conv._id) e.currentTarget.style.background='rgba(0,0,0,0.03)'; }}
                          onMouseLeave={(e) => { setHovConv(null); if(activeConvId!==conv._id) e.currentTarget.style.background='transparent'; }}
                          onClick={() => onSelectConv(conv._id)}>
                          <span className="flex-1 px-3 py-2 text-[13px] truncate"
                            style={{ color: activeConvId===conv._id ? '#4457F5' : '#374151' }}>
                            {conv.title}
                          </span>
                          {hovConv===conv._id && (
                            <button onClick={(e) => { e.stopPropagation(); onDeleteConv(conv._id); }}
                              className="flex-shrink-0 mr-2 p-1 rounded transition-colors"
                              style={{ color:'#D1D5DB' }}
                              onMouseEnter={(e) => e.currentTarget.style.color='#DC2626'}
                              onMouseLeave={(e) => e.currentTarget.style.color='#D1D5DB'}>
                              <Ico d={D.trash} size={12} />
                            </button>
                          )}
                        </div>
                      ))}
                    </div>
                  )
                )
              )}
            </div>

            {/* User profile / footer */}
            <div className="flex-shrink-0 p-3" style={{ borderTop:'1px solid #E5E7EB', position:'relative' }}>
              {user ? (
                <>
                  <button onClick={() => setShowUserMenu(!showUserMenu)}
                    className="w-full flex items-center gap-2.5 px-2 py-1.5 rounded-lg cursor-pointer transition-all" style={{ background: showUserMenu ? '#F3F4F6' : 'transparent' }}>
                    <div className="w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center text-[11px] font-bold text-white"
                      style={{ background:'linear-gradient(135deg,#5B6CF9,#4457F5)' }}>
                      {user.name.charAt(0).toUpperCase()}
                    </div>
                    <div className="flex-1 min-w-0 text-left">
                      <p className="text-[12px] font-semibold text-[#111827] truncate">{user.name}</p>
                      <p className="text-[10px] truncate" style={{ color:'#9CA3AF' }}>{user.email}</p>
                    </div>
                  </button>
                  
                  {showUserMenu && (
                    <div className="absolute bottom-full left-3 right-3 mb-2 rounded-xl overflow-hidden" style={{ background:'#FFFFFF', boxShadow:'0 20px 60px rgba(0,0,0,0.12)', zIndex:50, minWidth:280 }}>
                      {/* User info header */}
                      <div className="px-4 py-3 border-b" style={{ borderColor:'#E5E7EB' }}>
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-full flex items-center justify-center text-[13px] font-bold text-white" style={{ background:'linear-gradient(135deg,#5B6CF9,#4457F5)' }}>
                            {user.name.charAt(0).toUpperCase()}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-[13px] font-semibold text-[#111827] truncate">{user.name}</p>
                            <p className="text-[12px] truncate" style={{ color:'#6B7280' }}>{user.email}</p>
                          </div>
                          <Ico d={D.close} size={16} className="cursor-pointer" style={{color:'#9CA3AF'}} onClick={() => setShowUserMenu(false)} />
                        </div>
                      </div>
                      
                      {/* Menu items */}
                      <div className="py-1">
                        <button className="w-full flex items-center gap-3 px-4 py-2.5 text-[13px] text-[#374151] hover:bg-[#F3F4F6] transition-colors"
                          style={{ background:'transparent' }}>
                          <Ico d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z' size={16} />
                          Profile
                        </button>
                        <button className="w-full flex items-center gap-3 px-4 py-2.5 text-[13px] text-[#374151] hover:bg-[#F3F4F6] transition-colors"
                          style={{ background:'transparent' }}>
                          <Ico d='M19.14 12.94c.4-1.16.74-2.36.9-3.54h-2.16a5.884 5.884 0 0 1-1.51 3.54h2.77zM15.5 5.86a5.884 5.884 0 0 1 1.51 3.54h2.77c-.16-1.18-.5-2.38-.9-3.54h-2.38zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z' size={16} />
                          Settings
                        </button>
                        <button className="w-full flex items-center gap-3 px-4 py-2.5 text-[13px] text-[#374151] hover:bg-[#F3F4F6] transition-colors"
                          style={{ background:'transparent' }}>
                          <Ico d='M11 18h2v-2h-2v2zm1-16C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-14c-2.21 0-4 1.79-4 4h2c0-1.1.9-2 2-2s2 .9 2 2c0 2-3 1.75-3 5h2c0-2.25 3-2.5 3-5 0-2.21-1.79-4-4-4z' size={16} />
                          Help
                        </button>
                      </div>
                      
                      {/* Divider */}
                      <div style={{ height:'1px', background:'#E5E7EB' }} />
                      
                      {/* Log out */}
                      <button onClick={() => { setShowUserMenu(false); onLogout(); }}
                        className="w-full flex items-center gap-3 px-4 py-2.5 text-[13px] text-[#DC2626] hover:bg-[#F3F4F6] transition-colors" style={{ background:'transparent' }}>
                        <Ico d={D.logout} size={16} />
                        Log out
                      </button>
                    </div>
                  )}
                </>
              ) : (
                <p className="text-center text-[10px]" style={{ color:'#9CA3AF' }}>
                  Aroha · India Idsp Surveillance
                </p>
              )}
            </div>
          </div>
        )}
      </aside>
    </>
  );
}

// ── Chat Navbar ───────────────────────────────────────────────────
function ChatNavbar({ onToggleSidebar, sidebarOpen, onClear, onExport, count, user, onShowAuth, onToggleSidebarClose }) {
  const [hov, setHov] = useState(null);
  const btns = [
    { id:'exp', icon:D.export, tip:'Export', fn:onExport },
    { id:'clr', icon:D.trash,  tip:'Clear',  fn:onClear  },
  ];

  return (
    <header className="relative flex-shrink-0 flex items-center justify-between z-10"
      style={{ height:54, padding:'0 14px 0 12px', background:'#FFFFFF', borderBottom:'1px solid #E5E7EB' }}>
      <div className="absolute top-0 left-0 right-0 h-[2px]" style={{ background:'#E8002D' }} />

      <div className="flex items-center gap-2">
        <button onClick={() => sidebarOpen ? onToggleSidebarClose() : onToggleSidebar()}
          className="w-8 h-8 flex items-center justify-center rounded-lg transition-all"
          style={{ color:'#9CA3AF', background:'transparent' }}
          onMouseEnter={(e) => { e.currentTarget.style.background='#F3F4F6'; e.currentTarget.style.color='#374151'; }}
          onMouseLeave={(e) => { e.currentTarget.style.background='transparent'; e.currentTarget.style.color='#9CA3AF'; }}>
          <Ico d={sidebarOpen ? D.close : D.menu} size={18} />
        </button>
      </div>

      <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-[7px]">
        <span className="inline-block w-[5px] h-[5px] rounded-full"
          style={{ background:'#0D9488', boxShadow:'0 0 5px #0D9488', animation:'chatPulse 2s infinite' }} />
        <span className="text-[11px] font-medium hidden sm:inline" style={{ color:'#9CA3AF', letterSpacing:'.04em' }}>Surveillance active</span>
      </div>

      {/* Right actions */}
      <div className="flex items-center gap-1.5">
        {count > 0 && btns.map((b) => (
          <button key={b.id} onClick={b.fn} title={b.tip}
            onMouseEnter={() => setHov(b.id)} onMouseLeave={() => setHov(null)}
            className="w-[28px] h-[28px] rounded-[5px] flex items-center justify-center cursor-pointer transition-all duration-150"
            style={{
              background: hov===b.id?'#F3F4F6':'#F9FAFB',
              border: '1px solid #E5E7EB',
              color: hov===b.id?'#374151':'#9CA3AF',
            }}>
            <Ico d={b.icon} size={13} />
          </button>
        ))}
        {!user && (
          <button onClick={() => onShowAuth('login')}
            className="ml-1 px-3 py-1.5 rounded-lg text-[12px] font-semibold transition-all"
            style={{ background:'linear-gradient(135deg,#5B6CF9,#4457F5)', color:'#fff', boxShadow:'0 2px 8px rgba(68,87,245,0.22)' }}>
            Sign in
          </button>
        )}
      </div>
    </header>
  );
}

// ── Main Chat Page ────────────────────────────────────────────────
function ChatPageInner() {
  const searchParams = useSearchParams();
  const router       = useRouter();

  const [user, setUser]               = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [authModal, setAuthModal]     = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId]   = useState(null);
  const [msgs, setMsgs]               = useState([]);
  const [loading, setLoad]            = useState(false);
  const bottomRef                     = useRef(null);

  useEffect(() => {
    const handleResize = () => {
      setSidebarOpen(window.innerWidth >= 1024);
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    const authParam = searchParams.get('auth');
    if (authParam === 'login' || authParam === 'signup') {
      setAuthModal(authParam);
      router.replace('/chat', { scroll: false });
    }
  }, [searchParams, router]);

  useEffect(() => {
    document.documentElement.style.height = '100%';
    document.body.classList.add('chat-page-body');
    return () => {
      document.body.classList.remove('chat-page-body');
      document.documentElement.style.height = '';
    };
  }, []);

  useEffect(() => {
    fetch('/api/auth/me')
      .then((r) => r.json())
      .then((d) => { if (d.user) { setUser(d.user); loadConversations(); } })
      .catch(() => {})
      .finally(() => setAuthChecked(true));
  }, []);

  useEffect(() => {
    const t = setTimeout(() => bottomRef.current?.scrollIntoView({ behavior:'smooth' }), 50);
    return () => clearTimeout(t);
  }, [msgs, loading]);

  const loadConversations = async () => {
    try {
      const r = await fetch('/api/conversations');
      if (r.ok) { const d = await r.json(); setConversations(d.conversations || []); }
    } catch {}
  };

  const handleAuth = (userData) => {
    setUser(userData);
    setAuthModal(null);
    loadConversations();
  };

  const handleLogout = async () => {
    await fetch('/api/auth/logout', { method:'POST' });
    setUser(null);
    setConversations([]);
    setActiveConvId(null);
    setMsgs([]);
  };

  const startNewChat = () => {
    setActiveConvId(null);
    setMsgs([]);
  };

  const selectConversation = async (id) => {
    setActiveConvId(id);
    setMsgs([]);
    try {
      const r = await fetch(`/api/conversations/${id}`);
      if (r.ok) {
        const d = await r.json();
        setMsgs(d.messages.map((m) => ({
          id: m._id, role: m.role, text: m.text, isError: m.isError,
        })));
      }
    } catch {}
  };

  const deleteConversation = async (id) => {
    await fetch(`/api/conversations/${id}`, { method:'DELETE' });
    if (activeConvId === id) { setActiveConvId(null); setMsgs([]); }
    setConversations((prev) => prev.filter((c) => c._id !== id));
  };

  const send = useCallback(async (text) => {
    setMsgs((p) => [...p, { id: Date.now(), role:'user', text }]);
    setLoad(true);

    let convId = activeConvId;

    if (user && !convId) {
      try {
        const r = await fetch('/api/conversations', {
          method:'POST',
          headers:{ 'Content-Type':'application/json' },
          body: JSON.stringify({ title: text.slice(0, 60) }),
        });
        if (r.ok) {
          const d = await r.json();
          convId = d.conversation.id;
          setActiveConvId(convId);
          setConversations((prev) => [{
            _id: convId, title: text.slice(0,60), updatedAt: new Date().toISOString()
          }, ...prev]);
        }
      } catch {}
    }

    try {
      const r = await fetch('/api/chat', {
        method:'POST',
        headers:{ 'Content-Type':'application/json' },
        body: JSON.stringify({ message:text, conversationId: convId }),
      });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const reply = await r.json();
      const botText = typeof reply==='string' ? reply : reply.error || JSON.stringify(reply);
      setMsgs((p) => [...p, { id: Date.now()+1, role:'bot', text:botText }]);
    } catch (e) {
      setMsgs((p) => [...p, { id:Date.now()+1, role:'bot', isError:true, text:`Server error: ${e.message}` }]);
    } finally {
      setLoad(false);
    }
  }, [activeConvId, user]);

  const exportChat = () => {
    const lines = msgs.map((m) => `${m.role==='user'?'You':'Aroha'}: ${m.text}`);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([lines.join('\n\n')], { type:'text/plain' }));
    a.download = `Aroha_${new Date().toISOString().slice(0,10)}.txt`;
    a.click();
  };

  return (
    <>
      <style>{`
        body.chat-page-body { background: #FFFFFF; }
        @keyframes chatFadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
        @keyframes chatFadeIn { from{opacity:0} to{opacity:1} }
        @keyframes chatPulse  { 0%,100%{opacity:1} 50%{opacity:.3} }
        @keyframes chatBounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-5px);opacity:1} }
        .chat-scroll::-webkit-scrollbar { width: 4px; }
        .chat-scroll::-webkit-scrollbar-track { background: transparent; }
        .chat-scroll::-webkit-scrollbar-thumb { background: #E5E7EB; border-radius: 2px; }
        .chat-scroll::-webkit-scrollbar-thumb:hover { background: #D1D5DB; }
        .md p { margin: 0 0 8px; color: #1F2937; }
        .md h2, .md h3 { color: #111827; font-weight: 600; margin: 14px 0 6px; }
        .md strong { color: #111827; }
        .md code { background: #F3F4F6; color: #374151; padding: 1px 5px; border-radius: 4px; font-size: 13px; }
        .md pre { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px; padding: 12px 14px; overflow-x: auto; margin: 8px 0; }
        .md pre code { background: none; padding: 0; }
        .md ul { margin: 6px 0 10px 20px; }
        .md li { color: #374151; margin-bottom: 3px; }
        .md table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 13px; }
        .md th { background: #F9FAFB; color: #374151; font-weight: 600; padding: 7px 10px; border: 1px solid #E5E7EB; text-align: left; }
        .md td { padding: 7px 10px; border: 1px solid #E5E7EB; color: #374151; }
        .md hr { border: none; border-top: 1px solid #E5E7EB; margin: 14px 0; }
      `}</style>

      {authModal && (
        <AuthModal defaultTab={authModal} onClose={() => setAuthModal(null)} onAuth={handleAuth} />
      )}

      <div className="flex" style={{ height:'100dvh', overflow:'hidden', background:'#FFFFFF' }}>
        <Sidebar
          open={sidebarOpen}
          user={user}
          conversations={conversations}
          activeConvId={activeConvId}
          onNewChat={startNewChat}
          onSelectConv={selectConversation}
          onDeleteConv={deleteConversation}
          onShowAuth={(tab) => setAuthModal(tab)}
          onLogout={handleLogout}
          onToggleSidebar={() => setSidebarOpen(false)}
        />

        <div className="flex-1 flex flex-col overflow-hidden" style={{ background:'#FFFFFF' }}>
          <ChatNavbar
            sidebarOpen={sidebarOpen}
            onToggleSidebar={() => setSidebarOpen((o) => !o)}
            onToggleSidebarClose={() => setSidebarOpen(false)}
            onClear={() => { setMsgs([]); setActiveConvId(null); }}
            onExport={exportChat}
            count={msgs.length}
            user={user}
            onShowAuth={(tab) => setAuthModal(tab)}
          />

          {/* Scroll area */}
          <div className="flex-1 overflow-y-auto relative z-[1] chat-scroll" style={{ background:'#FFFFFF' }}>
            <div className="flex flex-col min-h-full" style={{ maxWidth:760, margin:'0 auto', padding:'24px 18px 8px' }}>
              {msgs.length === 0 ? (
                <Welcome onChip={send} />
              ) : (
                <div className="flex flex-col gap-[22px]">
                  {msgs.map((m) => <Message key={m.id} msg={m} />)}
                  {loading && (
                    <div className="flex justify-start" style={{ animation:'chatFadeUp .25s ease' }}>
                      <Dots />
                    </div>
                  )}
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          </div>

          <InputBar onSend={send} disabled={loading} />
        </div>
      </div>
    </>
  );
}

export default function ChatPage() {
  return (
    <Suspense fallback={null}>
      <ChatPageInner />
    </Suspense>
  );
}