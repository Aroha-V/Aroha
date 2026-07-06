'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';

/* ── Animated counter ── */
function Counter({ target, suffix = '', duration = 2000 }) {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  const started = useRef(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !started.current) {
          started.current = true;
          const startTime = performance.now();
          const step = (now) => {
            const progress = Math.min((now - startTime) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            setCount(Math.floor(eased * target));
            if (progress < 1) requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
        }
      },
      { threshold: 0.5 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [target, duration]);

  return (
    <span ref={ref}>
      {count.toLocaleString()}
      {suffix}
    </span>
  );
}

/* ── Feature card ── */
function FeatureCard({ icon, title, description, accent, delay }) {
  return (
    <div
      className="animate-fade-up group relative rounded-2xl border border-slate-100 bg-slate-50/60 p-6 hover:border-slate-200 hover:-translate-y-1 hover:shadow-[0_12px_40px_rgba(0,0,0,0.04)] transition-all duration-300 cursor-default"
      style={{ animationDelay: delay }}
    >
      {/* Hover glow */}
      <div
        className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
        style={{ background: `radial-gradient(circle at 50% 0%, ${accent}0d 0%, transparent 70%)` }}
      />

      <div
        className="w-11 h-11 rounded-xl flex items-center justify-center mb-4 text-xl"
        style={{ background: `${accent}12`, border: `1px solid ${accent}25` }}
      >
        {icon}
      </div>

      <h3 className="text-[15px] font-semibold text-slate-900 mb-2">{title}</h3>
      <p className="text-[13.5px] text-slate-600 leading-[1.7]">{description}</p>
    </div>
  );
}

/* ── Stat card ── */
function StatCard({ value, suffix, label, sublabel, accent, delay }) {
  return (
    <div
      className="animate-fade-up rounded-2xl border border-slate-100 bg-slate-50/60 p-6 text-center hover:border-slate-200 hover:shadow-[0_12px_40px_rgba(0,0,0,0.03)] transition-all duration-300"
      style={{ animationDelay: delay }}
    >
      <div
        className="text-4xl font-black mb-1 font-display text-black"
      >
        <Counter target={value} suffix={suffix} />
      </div>
      <div className="text-[14px] font-semibold text-slate-900 mb-1">{label}</div>
      <div className="text-[12px] text-slate-500">{sublabel}</div>
    </div>
  );
}

/* ── Step card (How It Works) ── */
function StepCard({ number, title, description, delay }) {
  return (
    <div
      className="animate-fade-up flex gap-4"
      style={{ animationDelay: delay }}
    >
      <div className="flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center text-sm font-black text-white border border-[#5B6CF9]/40"
        style={{ background: 'linear-gradient(135deg, #5B6CF9, #4457F5)' }}
      >
        {number}
      </div>
      <div>
        <h3 className="text-[15px] font-semibold text-slate-900 mb-1">{title}</h3>
        <p className="text-[13.5px] text-slate-600 leading-[1.7]">{description}</p>
      </div>
    </div>
  );
}

/* ── Main page ── */
export default function HomePage() {
  const [heroVisible, setHeroVisible] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setHeroVisible(true), 80);
    return () => clearTimeout(t);
  }, []);

  const features = [
    {
      icon: '🦠',
      title: 'Disease outbreak tracking',
      description: 'Real-time monitoring of disease outbreaks across all Indian states. Get instant alerts on emerging threats from Idsp data.',
      accent: '#E8002D',
      delay: '0.1s',
    },
    {
      icon: '🤖',
      title: 'AI-powered analysis',
      description: 'Ask natural language questions about outbreaks, case counts, and mortality rates. Aroha understands complex epidemiological queries.',
      accent: '#5B6CF9',
      delay: '0.2s',
    },
    {
      icon: '📊',
      title: 'Trend intelligence',
      description: 'Visualize disease trends over time with intelligent data aggregation across districts, states, and national levels.',
      accent: '#0D9488',
      delay: '0.3s',
    },
    {
      icon: '⚡',
      title: 'Instant responses',
      description: 'Powered by advanced LLMs and vector search over live Idsp data — answers in seconds, not hours of manual research.',
      accent: '#D97706',
      delay: '0.4s',
    },
    {
      icon: '💾',
      title: 'Chat history',
      description: 'Save and revisit your surveillance sessions. Build a personal knowledge base of tracked diseases and regions over time.',
      accent: '#5B6CF9',
      delay: '0.5s',
    },
    {
      icon: '🔒',
      title: 'Secure & private',
      description: 'Enterprise-grade security with JWT authentication. Your queries and saved sessions are fully private and encrypted.',
      accent: '#0D9488',
      delay: '0.6s',
    },
  ];

  const stats = [
    { value: 28, suffix: '+', label: 'States covered',    sublabel: 'All union territories included', accent: '#5B6CF9',  delay: '0.1s' },
    { value: 500, suffix: 'K+', label: 'Idsp records',    sublabel: 'Weekly surveillance reports', accent: '#E8002D',  delay: '0.2s' },
    { value: 50,  suffix: '+',  label: 'Diseases tracked', sublabel: 'Communicable & vector-borne',  accent: '#0D9488',  delay: '0.3s' },
    { value: 99,  suffix: '%',  label: 'Uptime SLA',       sublabel: 'Always-on surveillance',       accent: '#D97706',  delay: '0.4s' },
  ];

  const steps = [
    {
      number: '1',
      title: 'Create your account',
      description: 'Sign up in seconds with email. Your account is linked to a secure MongoDB-backed session with full chat history.',
      delay: '0.1s',
    },
    {
      number: '2',
      title: 'Ask anything about outbreaks',
      description: 'Type natural-language questions: "Dengue cases in Kerala last month?" or "Which states had cholera outbreaks in 2024?"',
      delay: '0.25s',
    },
    {
      number: '3',
      title: 'Get instant AI insights',
      description: 'AROHA queries the IDSP vector database, retrieves relevant surveillance data, and delivers a structured, cited response.',
      delay: '0.4s',
    },
    {
      number: '4',
      title: 'Save & revisit sessions',
      description: 'All conversations are saved to your MongoDB profile. Export reports, share insights, and build your disease intelligence history.',
      delay: '0.55s',
    },
  ];

  return (
    <main className="relative overflow-x-hidden">
      {/* ═══════════════════════════════════════════════════════════
          HERO SECTION
      ═══════════════════════════════════════════════════════════ */}
      <section className="relative min-h-screen flex flex-col items-center justify-center text-center px-6 pt-24 pb-20 overflow-hidden">
        {/* Background orbs */}
        <div
          className="absolute top-[-20%] left-[50%] -translate-x-1/2 w-[900px] h-[600px] rounded-full opacity-10 pointer-events-none animate-[orbDrift_14s_ease-in-out_infinite]"
          style={{ background: 'radial-gradient(ellipse, rgba(91,108,249,0.15) 0%, transparent 70%)', filter: 'blur(60px)' }}
        />
        <div
          className="absolute bottom-0 right-[-10%] w-[500px] h-[500px] rounded-full opacity-[0.08] pointer-events-none"
          style={{ background: 'radial-gradient(circle, rgba(232,0,45,0.2) 0%, transparent 70%)', filter: 'blur(80px)' }}
        />
        <div
          className="absolute bottom-[20%] left-[-5%] w-[400px] h-[400px] rounded-full opacity-[0.06] pointer-events-none"
          style={{ background: 'radial-gradient(circle, rgba(0,229,204,0.15) 0%, transparent 70%)', filter: 'blur(80px)' }}
        />

        {/* Grid overlay */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            backgroundImage: `
              linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px),
              linear-gradient(90deg, rgba(0,0,0,0.03) 1px, transparent 1px)
            `,
            backgroundSize: '60px 60px',
            maskImage: 'radial-gradient(ellipse 80% 80% at 50% 50%, black 40%, transparent 100%)',
            WebkitMaskImage: 'radial-gradient(ellipse 80% 80% at 50% 50%, black 40%, transparent 100%)',
          }}
        />

        {/* Content */}
        <div className="relative z-10 max-w-4xl mx-auto">
          {/* Badge */}
          <div
            className={`inline-flex items-center gap-2 px-4 py-2 rounded-full border text-[12px] font-semibold mb-8 transition-all duration-700 ${
              heroVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
            style={{
              background: 'rgba(232,0,45,0.04)',
              borderColor: 'rgba(232,0,45,0.15)',
              color: '#d01c3c',
            }}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-[#ff1744] animate-pulse" />
            India Idsp disease surveillance · Powered by Ai
          </div>

          {/* Headline */}
          <h1
            className={`font-display font-black leading-[1.08] tracking-tight mb-6 transition-all duration-700 delay-100 ${
              heroVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
            }`}
            style={{ fontSize: 'clamp(2.2rem, 5vw, 3.8rem)' }}
          >
            <span className="text-slate-900">Disease intelligence</span>
            <br />
            <span className="text-black">
              at your fingertips
            </span>
          </h1>

          {/* Subheadline */}
          <p
            className={`text-[1.1rem] leading-[1.75] text-slate-600 max-w-2xl mx-auto mb-10 transition-all duration-700 delay-200 ${
              heroVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
            }`}
          >
            Ask plain English questions about disease outbreaks, case counts, and mortality trends across all Indian states. Powered by real Idsp surveillance data and cutting-edge Ai.
          </p>

          {/* CTA buttons */}
          <div
            className={`flex flex-wrap items-center justify-center gap-4 mb-14 transition-all duration-700 delay-300 ${
              heroVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
            }`}
          >
            <Link
              href="/chat"
              className="relative group flex items-center gap-2.5 px-10 py-4 text-[16px] font-bold text-white rounded-xl overflow-hidden transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_16px_48px_rgba(91,108,249,0.25)]"
              style={{
                background: 'linear-gradient(135deg, #5B6CF9 0%, #4457F5 100%)',
                boxShadow: '0 4px 28px rgba(91,108,249,0.15)',
              }}
            >
              <span className="absolute inset-0 bg-white/0 group-hover:bg-white/[0.07] transition-all duration-300" />
              <span className="relative z-10">Get started free</span>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" className="relative z-10">
                <path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8z" />
              </svg>
            </Link>

            <a
              href="#how-it-works"
              className="flex items-center gap-2 px-7 py-4 text-[15px] font-semibold text-slate-600 rounded-xl border border-slate-200 hover:text-slate-900 hover:border-slate-300 hover:bg-slate-50 transition-all duration-200"
            >
              See how it works
              <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                <path d="M7 10l5 5 5-5z" />
              </svg>
            </a>
          </div>

          {/* Sample queries */}
          <div
            className={`transition-all duration-700 delay-500 ${
              heroVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            <p className="text-[11.5px] font-bold text-slate-400 tracking-[0.06em] mb-3">
              Try asking:
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {[
                'Dengue outbreaks in Tamil Nadu',
                'Cholera cases in Maharashtra',
                'Top diseases in Kerala 2024',
                'Malaria deaths by state',
                'Recent Idsp alerts',
              ].map((q) => (
                <span
                  key={q}
                  className="px-3.5 py-1.5 text-[12.5px] text-slate-600 bg-slate-50 border border-slate-200 rounded-lg hover:border-[#5B6CF9]/40 hover:text-slate-900 hover:bg-slate-100/50 transition-all duration-150 cursor-pointer"
                >
                  {q}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div
          className={`absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 transition-all duration-700 delay-700 ${
            heroVisible ? 'opacity-100' : 'opacity-0'
          }`}
        >
          <span className="text-[10px] font-semibold text-slate-400 tracking-[0.12em]">
            Scroll
          </span>
          <div className="w-5 h-8 rounded-full border border-slate-200 flex items-start justify-center p-1">
            <div className="w-1 h-2 rounded-full bg-[#5B6CF9] animate-[float_1.8s_ease-in-out_infinite]" />
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════
          STATS SECTION
      ═══════════════════════════════════════════════════════════ */}
      <section id="stats" className="py-20 px-6">
        <div className="max-w-[1200px] mx-auto">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            {stats.map((s) => (
              <StatCard key={s.label} {...s} />
            ))}
          </div>
        </div>
      </section>

      {/* ─── divider ─── */}
      <div
        className="max-w-[1200px] mx-auto px-6 h-px"
        style={{
          background:
            'linear-gradient(90deg, transparent, rgba(0,0,0,0.06) 30%, rgba(91,108,249,0.2) 50%, rgba(0,0,0,0.06) 70%, transparent)',
        }}
      />

      {/* ═══════════════════════════════════════════════════════════
          FEATURES SECTION
      ═══════════════════════════════════════════════════════════ */}
      <section id="features" className="py-24 px-6">
        <div className="max-w-[1200px] mx-auto">
          {/* Section header */}
          <div className="text-center mb-14">
            <span className="inline-block text-[11.5px] font-bold text-[#4f46e5] tracking-[0.12em] mb-3">
              Platform features
            </span>
            <h2 className="text-[clamp(1.8rem,4vw,2.8rem)] font-black text-slate-900 font-display leading-tight mb-4">
              Everything you need for<br />
              <span className="text-black">
                intelligent surveillance
              </span>
            </h2>
            <p className="text-[15px] text-slate-600 max-w-xl mx-auto leading-[1.75]">
              Aroha combines India's largest disease surveillance database with state-of-the-art Ai to give you unparalleled epidemiological insights.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((f) => (
              <FeatureCard key={f.title} {...f} />
            ))}
          </div>
        </div>
      </section>

      {/* ─── divider ─── */}
      <div
        className="max-w-[1200px] mx-auto px-6 h-px"
        style={{
          background:
            'linear-gradient(90deg, transparent, rgba(0,0,0,0.06) 30%, rgba(0,229,204,0.2) 50%, rgba(0,0,0,0.06) 70%, transparent)',
        }}
      />

      {/* ═══════════════════════════════════════════════════════════
          HOW IT WORKS
      ═══════════════════════════════════════════════════════════ */}
      <section id="how-it-works" className="py-24 px-6">
        <div className="max-w-[1100px] mx-auto">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            {/* Left — steps */}
            <div>
              <span className="inline-block text-[11.5px] font-bold text-[#0d9488] tracking-[0.12em] mb-4">
                How it works
              </span>
              <h2 className="text-[clamp(1.8rem,4vw,2.8rem)] font-black text-slate-900 font-display leading-tight mb-4">
                From question to<br />insight in seconds
              </h2>
              <p className="text-[15px] text-slate-600 leading-[1.75] mb-10">
                Aroha makes complex epidemiological data accessible to everyone — from public health officials to curious citizens.
              </p>
              <div className="flex flex-col gap-8">
                {steps.map((s) => (
                  <StepCard key={s.number} {...s} />
                ))}
              </div>
            </div>

            {/* Right — chat preview card */}
            <div className="relative">
              {/* Glow */}
              <div
                className="absolute -inset-10 rounded-3xl opacity-30 pointer-events-none"
                style={{
                  background: 'radial-gradient(ellipse, rgba(91,108,249,0.15) 0%, transparent 70%)',
                  filter: 'blur(30px)',
                }}
              />

              {/* Chat mockup */}
              <div className="relative rounded-2xl border border-slate-200 bg-slate-50/80 overflow-hidden shadow-[0_24px_80px_rgba(0,0,0,0.06)]">
                {/* Chat header */}
                <div className="flex items-center gap-3 px-5 py-3.5 border-b border-slate-200/60 bg-white">
                  <div className="flex gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-[#E8002D]/60" />
                    <div className="w-2.5 h-2.5 rounded-full bg-[#F59E0B]/60" />
                    <div className="w-2.5 h-2.5 rounded-full bg-[#00E5CC]/60" />
                  </div>
                  <span className="text-[11px] font-semibold text-slate-500 tracking-[0.08em]">
                    Aroha chat
                  </span>
                  <span className="ml-auto flex items-center gap-1.5 text-[10px] text-[#0D9488]">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#0d9488] animate-pulse" />
                    Live
                  </span>
                </div>

                {/* Messages */}
                <div className="p-5 flex flex-col gap-4 min-h-[320px]">
                  {/* User message */}
                  <div className="flex justify-end gap-2.5">
                    <div
                      className="max-w-[80%] px-4 py-2.5 rounded-[10px_2px_10px_10px] text-[13px] text-white leading-[1.65]"
                      style={{
                        background: 'linear-gradient(135deg, #5B6CF9, #4457F5)',
                        boxShadow: '0 4px 16px rgba(91,108,249,0.2)',
                      }}
                    >
                      Dengue outbreaks in Tamil Nadu last month?
                    </div>
                    <div className="w-7 h-7 rounded-lg flex-shrink-0 bg-slate-100 border border-slate-200 flex items-center justify-center text-xs">
                      👤
                    </div>
                  </div>

                  {/* Bot response */}
                  <div className="flex gap-2.5">
                    <div className="w-7 h-7 rounded-lg flex-shrink-0 bg-slate-100 border border-slate-200 flex items-center justify-center">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                        <path d="M12 2L2 7l10 5 10-5-10-5z" fill="#E8002D" opacity="0.9" />
                        <path d="M2 17l10 5 10-5" stroke="#5B6CF9" strokeWidth="1.5" fill="none" />
                        <path d="M2 12l10 5 10-5" stroke="#64748B" strokeWidth="1.5" fill="none" />
                      </svg>
                    </div>
                    <div
                      className="max-w-[85%] px-4 py-3 rounded-[2px_10px_10px_10px] text-[13px] text-slate-800 leading-[1.7] border border-slate-100 border-l-2"
                      style={{
                        background: '#ffffff',
                        borderLeftColor: '#5B6CF9',
                        boxShadow: '0 2px 12px rgba(0,0,0,0.02)',
                      }}
                    >
                      <p className="mb-2">
                        <strong className="text-slate-900 font-semibold">Tamil Nadu dengue alert (Idsp report)</strong>
                      </p>
                      <p className="text-slate-500 mb-1.5">📍 <span className="text-slate-800 font-medium">Chennai</span> — 1,247 cases, 3 deaths</p>
                      <p className="text-slate-500 mb-1.5">📍 <span className="text-slate-800 font-medium">Coimbatore</span> — 834 cases, 1 death</p>
                      <p className="text-slate-500 mb-1.5">📍 <span className="text-slate-800 font-medium">Madurai</span> — 612 cases, 0 deaths</p>
                      <p className="mt-2 text-[12px] text-slate-400">Source: Idsp weekly report · Week 21, 2025</p>
                    </div>
                  </div>

                  {/* Typing dots */}
                  <div className="flex gap-2.5">
                    <div className="w-7 h-7 rounded-lg flex-shrink-0 bg-slate-100 border border-slate-200 flex items-center justify-center">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                        <path d="M12 2L2 7l10 5 10-5-10-5z" fill="#E8002D" opacity="0.9" />
                      </svg>
                    </div>
                    <div className="px-4 py-3 bg-white border border-slate-100 border-l-2 border-[#5B6CF9] rounded-[2px_10px_10px_10px] flex gap-1 items-center">
                      {[0, 1, 2].map((i) => (
                        <span
                          key={i}
                          className="w-1.5 h-1.5 rounded-full bg-[#5B6CF9] opacity-60 animate-pulse"
                          style={{ animationDelay: `${i * 0.18}s` }}
                        />
                      ))}
                    </div>
                  </div>
                </div>

                {/* Input bar */}
                <div className="px-5 pb-5">
                  <div className="flex items-center gap-2.5 bg-white border border-slate-200 rounded-xl px-4 py-3 shadow-sm">
                    <span className="text-[13px] text-slate-400 flex-1">Ask about outbreaks, states, diseases…</span>
                    <button
                      className="w-7 h-7 rounded-lg flex items-center justify-center text-white flex-shrink-0"
                      style={{ background: 'linear-gradient(135deg, #5B6CF9, #4457F5)' }}
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ─── divider ─── */}
      <div
        className="max-w-[1200px] mx-auto px-6 h-px"
        style={{
          background:
            'linear-gradient(90deg, transparent, rgba(0,0,0,0.06) 30%, rgba(232,0,45,0.2) 50%, rgba(0,0,0,0.06) 70%, transparent)',
        }}
      />

      {/* ═══════════════════════════════════════════════════════════
          ABOUT / TESTIMONIAL STRIP
      ═══════════════════════════════════════════════════════════ */}
      <section id="about" className="py-20 px-6">
        <div className="max-w-[900px] mx-auto">
          <div className="relative rounded-3xl border border-slate-200/80 bg-slate-50/30 overflow-hidden p-10 text-center">
            {/* BG glow */}
            <div
              className="absolute inset-0 pointer-events-none"
              style={{ background: 'radial-gradient(ellipse 70% 60% at 50% 50%, rgba(91,108,249,0.04) 0%, transparent 70%)' }}
            />

            <div className="relative z-10">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-[#5B6CF9]/20 bg-[#5B6CF9]/5 text-[11.5px] font-semibold text-[#5B6CF9] tracking-[0.1em] mb-6">
                Our mission
              </div>
              <h2 className="text-[clamp(1.6rem,3.5vw,2.5rem)] font-black text-slate-900 font-display leading-tight mb-5">
                Making India's disease data<br />accessible to everyone
              </h2>
              <p className="text-[15px] text-slate-600 max-w-xl mx-auto leading-[1.8] mb-8">
                Aroha was built to democratize access to India's Idsp surveillance data. Whether you're a public health official, researcher, journalist, or concerned citizen — you deserve real-time, Ai-curated disease intelligence.
              </p>

              {/* Attribute chips */}
              <div className="flex flex-wrap justify-center gap-3">
                {[
                  { icon: '🏥', label: 'Idsp data source' },
                  { icon: '🤖', label: 'Gemini Ai' },
                  { icon: '🗄️', label: 'MongoDB backend' },
                  { icon: '⚡', label: 'Next.js 14' },
                  { icon: '🔐', label: 'JWT auth' },
                ].map(({ icon, label }) => (
                  <div
                    key={label}
                    className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-white border border-slate-200 text-[13px] text-slate-600 hover:text-slate-900 hover:border-slate-300 shadow-sm transition-all duration-150"
                  >
                    <span>{icon}</span>
                    <span>{label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════
          FINAL CTA SECTION
      ═══════════════════════════════════════════════════════════ */}
      <section className="py-24 px-6">
        <div className="max-w-[820px] mx-auto text-center relative">
          {/* Glow */}
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              background: 'radial-gradient(ellipse 80% 80% at 50% 50%, rgba(91,108,249,0.08) 0%, transparent 70%)',
              filter: 'blur(20px)',
            }}
          />

          <div className="relative z-10">
            <span className="inline-block text-[11.5px] font-bold text-[#5B6CF9] tracking-[0.12em] mb-4">
              Ready to start?
            </span>
            <h2 className="text-[clamp(2rem,5vw,3.5rem)] font-black text-slate-900 font-display leading-[1.08] tracking-tight mb-5">
              Start tracking outbreaks
              <br />
              <span className="text-black">
                in minutes
              </span>
            </h2>
            <p className="text-[16px] text-slate-600 leading-[1.75] mb-10 max-w-lg mx-auto">
              Join Aroha and get instant Ai-powered answers to India's most pressing disease surveillance questions. Free to get started.
            </p>

            <div className="flex flex-wrap items-center justify-center gap-4">
              <Link
                href="/chat"
                className="relative group flex items-center gap-2.5 px-10 py-4 text-[16px] font-bold text-white rounded-xl overflow-hidden transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_16px_48px_rgba(91,108,249,0.25)]"
                style={{
                  background: 'linear-gradient(135deg, #5B6CF9 0%, #4457F5 100%)',
                  boxShadow: '0 4px 28px rgba(91,108,249,0.15)',
                }}
              >
                <span className="absolute inset-0 bg-white/0 group-hover:bg-white/[0.07] transition-all duration-300" />
                <span className="relative z-10">Launch Aroha chat</span>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" className="relative z-10">
                  <path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8z" />
                </svg>
              </Link>

              <Link
                href="/chat"
                className="flex items-center gap-2 px-8 py-4 text-[16px] font-semibold text-slate-600 rounded-xl border border-slate-200 hover:text-slate-900 hover:border-slate-300 hover:bg-slate-50 transition-all duration-200"
              >
                Open chat model →
              </Link>
            </div>

            {/* Trust indicators */}
            <div className="flex flex-wrap justify-center gap-6 mt-10 text-[12px] text-slate-400">
              {[
                { icon: '✓', text: 'No credit card required' },
                { icon: '✓', text: 'Free forever tier' },
                { icon: '✓', text: 'Real Idsp data' },
                { icon: '✓', text: 'Powered by Gemini Ai' },
              ].map(({ icon, text }) => (
                <span key={text} className="flex items-center gap-1.5">
                  <span className="text-[#0d9488]">{icon}</span>
                  {text}
                </span>
              ))}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
