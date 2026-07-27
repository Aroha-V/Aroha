import './globals.css';

export const metadata = {
  title: 'AROHA — AI Disease Surveillance Intelligence',
  description:
    "India's most advanced AI-powered disease surveillance platform. Real-time insights from IDSP data — outbreaks, trends, and alerts across all states.",
  keywords: 'disease surveillance, IDSP, India health, outbreak monitoring, AI health intelligence',
  openGraph: {
    title: 'AROHA — AI Disease Surveillance Intelligence',
    description: "Real-time AI-powered insights from India's IDSP disease surveillance dataset.",
    type: 'website',
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="dark">
      <head>
        {/* Apply saved theme before first paint to avoid flash */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var saved = localStorage.getItem('theme');
                  var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                  var isDark = saved ? saved === 'dark' : true;
                  document.documentElement.classList.toggle('dark', isDark);
                } catch(e) {}
              })();
            `,
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
