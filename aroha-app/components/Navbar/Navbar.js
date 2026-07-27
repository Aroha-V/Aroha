'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';

const NAV_LINKS = [
  { href: '#features',     label: 'Features'     },
  { href: '#how-it-works', label: 'How it works' },
  { href: '/monitor',      label: 'Monitor'      },
  { href: '#stats',        label: 'Data'         },
  { href: '#about',        label: 'About'        },
];

/* ── Sun icon ── */
function SunIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1"  x2="12" y2="3"  />
      <line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22"  x2="5.64" y2="5.64"  />
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1"  y1="12" x2="3"  y2="12" />
      <line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
      <line x1="18.36" y1="5.64"  x2="19.78" y2="4.22"  />
    </svg>
  );
}

/* ── Moon icon ── */
function MoonIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
      <path d="M21 12.79A9 9 0 1111.21 3a7 7 0 009.79 9.79z" />
    </svg>
  );
}

export default function Navbar() {
  const [scrolled,  setScrolled]  = useState(false);
  const [menuOpen,  setMenuOpen]  = useState(false);
  const [darkMode,  setDarkMode]  = useState(false);
  const [mounted,   setMounted]   = useState(false);

  /* Restore saved theme on mount */
  useEffect(() => {
    setMounted(true);
    const saved = localStorage.getItem('theme');
    const isDark = saved ? saved === 'dark' : true; // default: dark
    setDarkMode(isDark);
    document.documentElement.classList.toggle('dark', isDark);
  }, []);

  /* Scroll listener */
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  /* Toggle handler */
  const toggleTheme = () => {
    const next = !darkMode;
    setDarkMode(next);
    document.documentElement.classList.toggle('dark', next);
    localStorage.setItem('theme', next ? 'dark' : 'light');
  };

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 h-[68px] transition-all duration-300 ${
        scrolled
          ? 'backdrop-blur-2xl'
          : ''
      }`}
      style={{
        background: scrolled ? 'rgba(33,33,33,0.95)' : 'transparent',
        borderBottom: scrolled ? '1px solid rgba(255,255,255,0.08)' : 'none',
        boxShadow: scrolled ? '0 4px 24px rgba(0,0,0,0.4)' : 'none',
      }}
    >
      {/* ── Animated top gradient bar ── */}
      <div
        className="absolute top-0 left-0 right-0 h-[2px]"
        style={{
          background: 'linear-gradient(90deg, #E8002D, #5B6CF9 50%, #00E5CC)',
          backgroundSize: '200% auto',
          animation: 'borderFlow 4s linear infinite',
        }}
      />

      <div className="max-w-[1200px] mx-auto px-6 h-full flex items-center justify-between gap-8 relative z-10">
        {/* ── Brand ── */}
        <Link href="/" className="flex items-center gap-2.5 group flex-shrink-0">
          <Image src="/aroha-logo.jpeg" alt="Aroha logo" width={36} height={36} className="rounded-[9px] object-cover" />
          <span className="text-base font-black tracking-[0.06em] font-display" style={{ color: '#ffffff' }}>
            Aroha
          </span>
          <span className="text-[9px] font-bold px-[7px] py-[2px] rounded-[4px] tracking-[0.1em]" style={{ color: '#E8002D', background: 'rgba(232,0,45,0.1)', border: '1px solid rgba(232,0,45,0.22)' }}>
            Idsp
          </span>
        </Link>

        {/* ── Desktop Nav Links ── */}
        <ul className="hidden md:flex items-center gap-1 list-none">
          {NAV_LINKS.map(({ href, label }) => (
            <li key={href}>
              <a
                href={href}
                className="block px-3.5 py-[7px] text-[13.5px] font-medium rounded-lg transition-all duration-150 tracking-[0.01em]"
                style={{ color: '#64748b' }}
                onMouseEnter={(e) => { e.currentTarget.style.color = '#e2e8f0'; e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = '#64748b'; e.currentTarget.style.background = 'transparent'; }}
              >
                {label}
              </a>
            </li>
          ))}
        </ul>

        {/* ── Right side: CTA + Theme toggler ── */}
        <div className="hidden md:flex items-center gap-2.5 flex-shrink-0">
          <Link
            href="/chat?auth=login"
            className="px-4 py-2 text-[13px] font-semibold rounded-lg transition-all duration-150"
            style={{ color: '#64748b', border: '1px solid rgba(255,255,255,0.1)', background: 'transparent' }}
            onMouseEnter={(e) => { e.currentTarget.style.color = '#e2e8f0'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.18)'; e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = '#64748b'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'; e.currentTarget.style.background = 'transparent'; }}
          >
            Sign in
          </Link>
          <Link
            href="/chat"
            className="flex items-center gap-1.5 px-4 py-2 text-[13px] font-semibold text-white rounded-lg transition-all duration-200 hover:-translate-y-[1px] hover:shadow-[0_8px_24px_rgba(91,108,249,0.35)]"
            style={{ background: 'linear-gradient(135deg, #5B6CF9 0%, #4457F5 100%)', boxShadow: '0 4px 16px rgba(91,108,249,0.2)' }}
          >
            Get started
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8z" />
            </svg>
          </Link>

          {/* ── Theme Toggle Button ── */}
          {mounted && (
            <button
              id="theme-toggle-btn"
              onClick={toggleTheme}
              aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              className="relative w-9 h-9 flex items-center justify-center rounded-lg transition-all duration-200 overflow-hidden"
              style={{
                background: darkMode ? '#2f2f2f' : '#F9FAFB',
                border: `1px solid ${darkMode ? 'rgba(255,255,255,0.1)' : '#E5E7EB'}`,
                color: darkMode ? '#fbbf24' : '#64748b',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = darkMode ? '#3a3a3a' : '#F3F4F6'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = darkMode ? '#2f2f2f' : '#F9FAFB'; }}
            >
              {/* Sun — visible in dark mode */}
              <span
                style={{
                  position: 'absolute',
                  transition: 'transform 0.3s ease, opacity 0.3s ease',
                  transform: darkMode ? 'rotate(0deg) scale(1)' : 'rotate(90deg) scale(0.5)',
                  opacity: darkMode ? 1 : 0,
                }}
              >
                <SunIcon />
              </span>
              {/* Moon — visible in light mode */}
              <span
                style={{
                  position: 'absolute',
                  transition: 'transform 0.3s ease, opacity 0.3s ease',
                  transform: darkMode ? 'rotate(-90deg) scale(0.5)' : 'rotate(0deg) scale(1)',
                  opacity: darkMode ? 0 : 1,
                }}
              >
                <MoonIcon />
              </span>
            </button>
          )}
        </div>

        {/* ── Mobile: Theme toggle + Hamburger ── */}
        <div className="md:hidden flex items-center gap-2 flex-shrink-0">
          {/* Mobile theme toggle */}
          {mounted && (
            <button
              id="theme-toggle-mobile-btn"
              onClick={toggleTheme}
              aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              className="relative w-8 h-8 flex items-center justify-center rounded-lg transition-all duration-200 overflow-hidden"
              style={{
                background: darkMode ? '#13131f' : '#F9FAFB',
                border: `1px solid ${darkMode ? 'rgba(255,255,255,0.1)' : '#E5E7EB'}`,
                color: darkMode ? '#fbbf24' : '#64748b',
              }}
            >
              <span
                style={{
                  position: 'absolute',
                  transition: 'transform 0.3s ease, opacity 0.3s ease',
                  transform: darkMode ? 'rotate(0deg) scale(1)' : 'rotate(90deg) scale(0.5)',
                  opacity: darkMode ? 1 : 0,
                }}
              >
                <SunIcon />
              </span>
              <span
                style={{
                  position: 'absolute',
                  transition: 'transform 0.3s ease, opacity 0.3s ease',
                  transform: darkMode ? 'rotate(-90deg) scale(0.5)' : 'rotate(0deg) scale(1)',
                  opacity: darkMode ? 0 : 1,
                }}
              >
                <MoonIcon />
              </span>
            </button>
          )}

          {/* Hamburger */}
          <button
            className="w-9 h-9 flex flex-col justify-center items-center gap-[5px] rounded-lg transition-all duration-150 flex-shrink-0"
            style={{ background: '#2f2f2f', border: '1px solid rgba(255,255,255,0.1)' }}
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label="Toggle menu"
            onMouseEnter={(e) => { e.currentTarget.style.background = '#3a3a3a'; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = '#2f2f2f'; }}
          >
            <span
              className={`block w-4 h-[1.5px] rounded-full transition-all duration-300 origin-center ${
                menuOpen ? 'translate-y-[6.5px] rotate-45' : ''
              }`}
              style={{ background: '#94a3b8' }}
            />
            <span
              className={`block w-4 h-[1.5px] rounded-full transition-all duration-300 ${
                menuOpen ? 'opacity-0 scale-x-0' : ''
              }`}
              style={{ background: '#94a3b8' }}
            />
            <span
              className={`block w-4 h-[1.5px] rounded-full transition-all duration-300 origin-center ${
                menuOpen ? '-translate-y-[6.5px] -rotate-45' : ''
              }`}
              style={{ background: '#94a3b8' }}
            />
          </button>
        </div>
      </div>

      {/* ── Mobile Drawer ── */}
      <div
        className={`md:hidden absolute top-full left-0 right-0 overflow-hidden transition-all duration-300 ${
          menuOpen ? 'max-h-[400px] opacity-100' : 'max-h-0 opacity-0'
        } backdrop-blur-2xl`}
        style={{ background: 'rgba(33,33,33,0.97)', borderBottom: '1px solid rgba(255,255,255,0.08)' }}
      >
        <div className="px-6 py-5">
          <ul className="flex flex-col gap-1 list-none mb-5">
            {NAV_LINKS.map(({ href, label }) => (
              <li key={href}>
                <a
                  href={href}
                  className="block px-4 py-3 text-[15px] font-medium rounded-lg transition-all duration-150"
                  style={{ color: '#64748b' }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = '#e2e8f0'; e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = '#64748b'; e.currentTarget.style.background = 'transparent'; }}
                  onClick={() => setMenuOpen(false)}
                >
                  {label}
                </a>
              </li>
            ))}
          </ul>
          <div className="flex gap-3 pt-4" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
            <Link
              href="/chat?auth=login"
              className="flex-1 text-center py-2.5 text-[14px] font-semibold rounded-lg transition-all duration-150"
              style={{ color: '#64748b', border: '1px solid rgba(255,255,255,0.1)', background: 'transparent' }}
              onClick={() => setMenuOpen(false)}
            >
              Sign in
            </Link>
            <Link
              href="/chat"
              className="flex-1 text-center py-2.5 text-[14px] font-semibold text-white rounded-lg"
              style={{ background: 'linear-gradient(135deg, #5B6CF9, #4457F5)' }}
              onClick={() => setMenuOpen(false)}
            >
              Get started
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}
