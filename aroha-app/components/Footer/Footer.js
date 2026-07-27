'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';

const FOOTER_LINKS = {
  Product: [
    { label: 'Features',     href: '#features'     },
    { label: 'How it works', href: '#how-it-works' },
    { label: 'Data sources', href: '#stats'        },
    { label: 'Changelog',    href: '#'             },
  ],
  Platform: [
    { label: 'Get started', href: '/chat' },
    { label: 'Sign in',     href: '/login'    },
    { label: 'Dashboard',   href: '/chat'     },
    { label: 'Api docs',    href: '#'         },
  ],
  'Data & privacy': [
    { label: 'Privacy policy', href: '#'                              },
    { label: 'Terms of use',   href: '#'                              },
    { label: 'Idsp portal',    href: 'https://idsp.mohfw.gov.in', external: true },
    { label: 'Data sources',   href: '#'                              },
  ],
};

export default function Footer() {
  const year = new Date().getFullYear();
  const [darkMode, setDarkMode] = useState(true);

  useEffect(() => {
    const sync = () => setDarkMode(document.documentElement.classList.contains('dark'));
    sync();
    const observer = new MutationObserver(sync);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  const bg      = darkMode ? '#212121' : '#ffffff';
  const cardBg  = darkMode ? '#2f2f2f' : '#f8fafc';
  const border  = darkMode ? 'rgba(255,255,255,0.08)' : '#e2e8f0';
  const textMain= darkMode ? '#ececec'  : '#0f172a';
  const textSub = darkMode ? '#8e8ea0'  : '#475569';
  const textMute= darkMode ? '#6b7280'  : '#94a3b8';

  return (
    <footer className="relative mt-20 transition-colors duration-300" style={{ background: bg, borderTop: `1px solid ${border}` }}>
      {/* Top gradient line */}
      <div className="h-px" style={{ background: 'linear-gradient(90deg, transparent, rgba(91,108,249,0.15) 50%, transparent)' }} />

      {/* Main grid */}
      <div className="max-w-[1200px] mx-auto px-6 pt-16 pb-12 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-[2fr_1fr_1fr_1fr] gap-12">
        {/* ── Brand column ── */}
        <div className="flex flex-col gap-4">
          <Link href="/" className="inline-flex items-center gap-2.5">
            <Image src="/aroha-logo.jpeg" alt="Aroha logo" width={40} height={40} className="rounded-[10px] object-cover" />
            <span className="text-lg font-black tracking-[0.06em]" style={{ color: '#ffffff' }}>Aroha</span>
          </Link>

          <p className="text-[13.5px] leading-[1.75] max-w-[270px]" style={{ color: '#475569' }}>
            India's Ai-powered disease surveillance intelligence platform — real-time insights from Idsp data across all states.
          </p>

          {/* Live status */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#0d9488]/[0.05] border border-[#0d9488]/15 w-fit">
            <span className="w-1.5 h-1.5 rounded-full bg-[#0d9488] animate-pulse-slow" />
            <span className="text-[11.5px] font-medium text-[#0d9488]">Surveillance active</span>
          </div>

          {/* Social icons */}
          <div className="flex gap-2 mt-1">
            {[
              {
                label: 'GitHub',
                href: '#',
                path: 'M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z',
              },
              {
                label: 'Twitter',
                href: '#',
                path: 'M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.742l7.737-8.835L1.254 2.25H8.08l4.259 5.631zm-1.161 17.52h1.833L7.084 4.126H5.117z',
              },
            ].map(({ label, href, path }) => (
              <a
                key={label}
                href={href}
                aria-label={label}
                className="w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-150 hover:-translate-y-0.5"
                style={{ background: '#13131f', border: '1px solid rgba(255,255,255,0.08)', color: '#334155' }}
                onMouseEnter={(e) => { e.currentTarget.style.color = '#94a3b8'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.14)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = '#334155'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)'; }}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                  <path d={path} />
                </svg>
              </a>
            ))}
          </div>
        </div>

        {/* ── Link columns ── */}
        {Object.entries(FOOTER_LINKS).map(([section, items]) => (
          <div key={section} className="flex flex-col gap-4">
            <h3 className="text-[11.5px] font-bold tracking-[0.08em]" style={{ color: '#94a3b8' }}>
              {section}
            </h3>
            <ul className="flex flex-col gap-2.5 list-none">
              {items.map(({ label, href, external }) => (
                <li key={label}>
                  <a
                    href={href}
                    target={external ? '_blank' : undefined}
                    rel={external ? 'noopener noreferrer' : undefined}
                    className="text-[13.5px] transition-colors duration-150"
                    style={{ color: '#334155' }}
                    onMouseEnter={(e) => { e.currentTarget.style.color = '#e2e8f0'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.color = '#334155'; }}
                  >
                    {label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* ── Bottom bar ── */}
      <div className="max-w-[1200px] mx-auto px-6 pb-7">
        <div className="h-px mb-6" style={{ background: 'rgba(255,255,255,0.06)' }} />
        <div className="flex flex-wrap items-center justify-between gap-4">
          <p className="text-[12.5px]" style={{ color: '#334155' }}>
            © {year} Aroha. Built with data from{' '}
            <a
              href="https://idsp.mohfw.gov.in"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#5B6CF9] hover:text-[#0d9488] transition-colors duration-150"
            >
              India Idsp
            </a>
            .
          </p>
          <div className="flex items-center gap-2 flex-wrap">
            {[
              { label: '● Live data',   color: 'text-[#ff1744] bg-[#E8002D]/10 border-[#E8002D]/25' },
              { label: 'Ai powered',    color: 'text-[#5B6CF9] bg-[#5B6CF9]/10 border-[#5B6CF9]/25' },
              { label: 'Open source',   color: 'text-[#0d9488] bg-[#0d9488]/10 border-[#0d9488]/25' },
            ].map(({ label, color }) => (
              <span
                key={label}
                className={`text-[10px] font-bold tracking-[0.06em] px-2.5 py-1 rounded-full border ${color}`}
              >
                {label}
              </span>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}
