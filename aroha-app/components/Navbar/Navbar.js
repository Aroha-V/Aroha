'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import dynamic from 'next/dynamic';

const NAV_LINKS = [
  { href: '#features',     label: 'Features'     },
  { href: '#how-it-works', label: 'How it works' },
  { href: '/monitor',      label: 'Monitor'      },
  { href: '#stats',        label: 'Data'         },
  { href: '#about',        label: 'About'        },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 h-[68px] transition-all duration-300 ${
        scrolled
          ? 'bg-white/90 backdrop-blur-2xl border-b border-slate-100 shadow-[0_4px_20px_rgba(0,0,0,0.03)]'
          : 'bg-transparent'
      }`}
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
          <span className="text-base font-black text-slate-900 tracking-[0.06em] font-display">
            Aroha
          </span>
          <span className="text-[9px] font-bold text-[#E8002D] bg-[#E8002D]/10 border border-[#E8002D]/25 px-[7px] py-[2px] rounded-[4px] tracking-[0.1em]">
            Idsp
          </span>
        </Link>

        {/* ── Desktop Nav Links ── */}
        <ul className="hidden md:flex items-center gap-1 list-none">
          {NAV_LINKS.map(({ href, label }) => (
            <li key={href}>
              <a
                href={href}
                className="block px-3.5 py-[7px] text-[13.5px] font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-50 rounded-lg transition-all duration-150 tracking-[0.01em]"
              >
                {label}
              </a>
            </li>
          ))}
        </ul>

        {/* ── CTA Buttons ── */}
        <div className="hidden md:flex items-center gap-2.5 flex-shrink-0">
          <Link
            href="/chat?auth=login"
            className="px-4 py-2 text-[13px] font-semibold text-slate-600 border border-slate-200 rounded-lg hover:text-slate-900 hover:border-slate-300 hover:bg-slate-50 transition-all duration-150"
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
        </div>

        {/* ── Mobile Hamburger ── */}
        <button
          className="md:hidden w-9 h-9 flex flex-col justify-center items-center gap-[5px] rounded-lg bg-slate-50 border border-slate-200 hover:bg-slate-100 transition-all duration-150 flex-shrink-0"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          <span
            className={`block w-4 h-[1.5px] bg-slate-600 rounded-full transition-all duration-300 origin-center ${
              menuOpen ? 'translate-y-[6.5px] rotate-45' : ''
            }`}
          />
          <span
            className={`block w-4 h-[1.5px] bg-slate-600 rounded-full transition-all duration-300 ${
              menuOpen ? 'opacity-0 scale-x-0' : ''
            }`}
          />
          <span
            className={`block w-4 h-[1.5px] bg-slate-600 rounded-full transition-all duration-300 origin-center ${
              menuOpen ? '-translate-y-[6.5px] -rotate-45' : ''
            }`}
          />
        </button>
      </div>

      {/* ── Mobile Drawer ── */}
      <div
        className={`md:hidden absolute top-full left-0 right-0 overflow-hidden transition-all duration-300 ${
          menuOpen ? 'max-h-[400px] opacity-100' : 'max-h-0 opacity-0'
        } bg-white/95 backdrop-blur-2xl border-b border-slate-100`}
      >
        <div className="px-6 py-5">
          <ul className="flex flex-col gap-1 list-none mb-5">
            {NAV_LINKS.map(({ href, label }) => (
              <li key={href}>
                <a
                  href={href}
                  className="block px-4 py-3 text-[15px] font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-50 rounded-lg transition-all duration-150"
                  onClick={() => setMenuOpen(false)}
                >
                  {label}
                </a>
              </li>
            ))}
          </ul>
          <div className="flex gap-3 pt-4 border-t border-slate-100">
            <Link
              href="/chat?auth=login"
              className="flex-1 text-center py-2.5 text-[14px] font-semibold text-slate-600 border border-slate-200 rounded-lg hover:text-slate-900 hover:border-slate-300 hover:bg-slate-50 transition-all duration-150"
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
