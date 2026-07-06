/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,jsx,ts,tsx}',
    './components/**/*.{js,jsx,ts,tsx}',
    './pages/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans:    ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        display: ['Space Grotesk', 'Inter', 'ui-sans-serif', 'sans-serif'],
        mono:    ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      colors: {
        bg: {
          DEFAULT: '#080810',
          2:       '#0d0d1a',
        },
        panel:   '#0f0f1e',
        surface: {
          DEFAULT: '#161628',
          2:       '#1e1e35',
          3:       '#262642',
        },
        brand: {
          red:    '#E8002D',
          red2:   '#ff1744',
          indigo: '#5B6CF9',
          indigo2:'#4457F5',
          indigo3:'#7c8fff',
          teal:   '#00E5CC',
          amber:  '#F59E0B',
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'hero-mesh':
          'radial-gradient(ellipse 80% 60% at 50% -20%, rgba(91,108,249,0.25) 0%, transparent 70%)',
        'hero-orb-1':
          'radial-gradient(circle at 20% 50%, rgba(91,108,249,0.18) 0%, transparent 60%)',
        'hero-orb-2':
          'radial-gradient(circle at 80% 50%, rgba(232,0,45,0.12) 0%, transparent 60%)',
      },
      boxShadow: {
        'indigo-glow': '0 4px 24px rgba(91,108,249,0.35)',
        'indigo-glow-lg': '0 8px 40px rgba(91,108,249,0.4)',
        'red-glow':    '0 4px 24px rgba(232,0,45,0.3)',
        'card':        '0 4px 24px rgba(0,0,0,0.3)',
        'card-hover':  '0 12px 40px rgba(0,0,0,0.45)',
      },
      animation: {
        'fade-in':    'fadeIn 0.4s ease both',
        'fade-up':    'fadeUp 0.6s cubic-bezier(0.22,1,0.36,1) both',
        'fade-up-slow':'fadeUp 0.8s cubic-bezier(0.22,1,0.36,1) both',
        'slide-left': 'slideLeft 0.6s cubic-bezier(0.22,1,0.36,1) both',
        'slide-right':'slideRight 0.6s cubic-bezier(0.22,1,0.36,1) both',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'float':      'float 6s ease-in-out infinite',
        'float-slow': 'float 9s ease-in-out infinite',
        'glow-pulse': 'glowPulse 2.5s ease-in-out infinite',
        'shimmer':    'shimmer 3s linear infinite',
        'orb-drift':  'orbDrift 12s ease-in-out infinite',
        'border-flow':'borderFlow 4s linear infinite',
        'spin-slow':  'spin 20s linear infinite',
        'counter':    'counter 0.5s ease both',
      },
      keyframes: {
        fadeIn:     { from:{opacity:'0'}, to:{opacity:'1'} },
        fadeUp:     { from:{opacity:'0',transform:'translateY(28px)'}, to:{opacity:'1',transform:'translateY(0)'} },
        slideLeft:  { from:{opacity:'0',transform:'translateX(-30px)'}, to:{opacity:'1',transform:'translateX(0)'} },
        slideRight: { from:{opacity:'0',transform:'translateX(30px)'}, to:{opacity:'1',transform:'translateX(0)'} },
        float:      { '0%,100%':{transform:'translateY(0)'}, '50%':{transform:'translateY(-14px)'} },
        glowPulse:  { '0%,100%':{boxShadow:'0 0 20px rgba(91,108,249,0.3)'}, '50%':{boxShadow:'0 0 50px rgba(91,108,249,0.6)'} },
        shimmer:    { '0%':{backgroundPosition:'-200% center'}, '100%':{backgroundPosition:'200% center'} },
        orbDrift:   { '0%,100%':{transform:'translate(0,0) scale(1)'}, '33%':{transform:'translate(30px,-20px) scale(1.05)'}, '66%':{transform:'translate(-20px,15px) scale(0.95)'} },
        borderFlow: { '0%':{backgroundPosition:'0% 50%'}, '50%':{backgroundPosition:'100% 50%'}, '100%':{backgroundPosition:'0% 50%'} },
      },
      transitionTimingFunction: {
        'spring': 'cubic-bezier(0.22, 1, 0.36, 1)',
      },
    },
  },
  plugins: [],
};
