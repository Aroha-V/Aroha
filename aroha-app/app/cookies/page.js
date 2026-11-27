import Navbar from '@/components/Navbar/Navbar';
import Footer from '@/components/Footer/Footer';

export default function CookiesPage() {
  return (
    <>
      <Navbar />

      <main className="min-h-screen">
        <section className="max-w-[900px] mx-auto px-6 pt-24 pb-24">
          <div className="max-w-[760px]">
            <p className="text-sm font-medium text-[#5B6CF9] mb-4">
              COOKIE POLICY
            </p>

            <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
              How Aroha uses cookies.
            </h1>

            <p className="mt-6 text-[15px] leading-7 text-slate-400">
              This page explains the browser storage used by Aroha to
              operate the platform.
            </p>

            <p className="mt-3 text-sm text-slate-500">
              Last updated: September 2026
            </p>
          </div>

          <div className="mt-14 space-y-10">

            <section>
              <h2 className="text-xl font-semibold">
                Authentication cookie
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Aroha uses an authentication cookie named
                <span className="text-slate-200"> aroha_token </span>
                to maintain an authenticated session after you sign in.
              </p>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                The cookie is HTTP-only, which means it is not directly
                accessible to client-side JavaScript. It is configured for
                the site and has a limited lifetime.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                Why this cookie is needed
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                The authentication cookie allows Aroha to recognize an
                authenticated user when accessing protected features such
                as saved conversations.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                Local browser storage
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Aroha also uses browser local storage for certain interface
                preferences, such as the selected theme. Local storage is
                different from cookies and is stored directly in your
                browser.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                Third-party services
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Some features use third-party services, including Firebase
                Authentication. Those services may use their own
                technologies and have their own privacy and cookie
                policies.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                Changes to this policy
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                As Aroha develops, the technologies used by the platform
                may change. This page will be updated when the web browser
                storage or cookies are used changes.
              </p>
            </section>

          </div>
        </section>
      </main>

      <Footer />
    </>
  );
}