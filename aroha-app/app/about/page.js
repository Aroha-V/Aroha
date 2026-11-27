import Navbar from '@/components/Navbar/Navbar';
import Footer from '@/components/Footer/Footer';

export default function AboutPage() {
  return (
    <>
      <Navbar />

      <main className="min-h-screen">
        {/* Hero */}
        <section className="max-w-[1100px] mx-auto px-6 pt-24 pb-20">
          <div className="max-w-[760px]">
            <p className="text-sm font-medium text-[#5B6CF9] mb-4">
              ABOUT AROHA
            </p>

            <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
              Listening before an outbreak spreads.
            </h1>

            <p className="mt-6 text-lg leading-8 text-slate-400">
              Aroha is an early-stage disease surveillance platform
              designed to detect unusual disease patterns from anonymous,
              neighbourhood-level symptom reports.
            </p>

            <div className="mt-6 inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-[#5B6CF9]/20 bg-[#5B6CF9]/10">
              <span className="w-1.5 h-1.5 rounded-full bg-[#5B6CF9]" />
              <span className="text-sm text-slate-300">
                Early stage · Actively building
              </span>
            </div>
          </div>
        </section>

        {/* Problem */}
        <section className="max-w-[1100px] mx-auto px-6 py-16">
          <div className="max-w-[760px]">
            <h2 className="text-2xl md:text-3xl font-semibold">
              The signals are often there.
            </h2>

            <p className="mt-5 text-[15px] leading-7 text-slate-400">
              Disease outbreaks can take time to reach formal reporting
              systems. Meanwhile, people in a neighbourhood may already
              be experiencing similar symptoms.
            </p>

            <p className="mt-4 text-[15px] leading-7 text-slate-400">
              Aroha is being built around a simple idea: listen to those
              signals earlier and look for patterns before they become
              larger outbreaks.
            </p>
          </div>
        </section>

        {/* How it works */}
        <section className="max-w-[1100px] mx-auto px-6 py-16">
          <h2 className="text-2xl md:text-3xl font-semibold">
            How it works
          </h2>

          <div className="grid md:grid-cols-4 gap-5 mt-10">
            <Step
              number="01"
              title="Report"
              description="A user anonymously reports symptoms through the app."
            />

            <Step
              number="02"
              title="Cluster"
              description="Reports are grouped by geography and symptom type."
            />

            <Step
              number="03"
              title="Detect"
              description="AI looks for unusual patterns against seasonal baselines."
            />

            <Step
              number="04"
              title="Alert"
              description="Potential signals can be surfaced to local health workers and officials."
            />
          </div>
        </section>

        {/* Features */}
        <section className="max-w-[1100px] mx-auto px-6 py-16">
          <h2 className="text-2xl md:text-3xl font-semibold">
            What we're building
          </h2>

          <div className="grid md:grid-cols-2 gap-5 mt-10">
            <Feature
              title="Neighbourhood-level signals"
              description="Hyperlocal symptom patterns at ward and mohalla level."
            />

            <Feature
              title="Anonymous reporting"
              description="The platform is designed around symptom signals rather than personal identity."
            />

            <Feature
              title="Anomaly detection"
              description="AI is used to identify unusual spikes against seasonal patterns."
            />

            <Feature
              title="Lightweight by design"
              description="Designed with low-data and low-smartphone environments in mind."
            />

            <Feature
              title="Local alerts"
              description="Designed to surface relevant signals to ASHA workers and local health officials."
            />
          </div>
        </section>

        {/* Status */}
        <section className="max-w-[1100px] mx-auto px-6 py-16 pb-24">
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-8 md:p-10">
            <p className="text-sm font-medium text-[#5B6CF9]">
              CURRENT STATUS
            </p>

            <h2 className="mt-3 text-2xl md:text-3xl font-semibold">
              We're still building.
            </h2>

            <p className="mt-4 max-w-[700px] text-[15px] leading-7 text-slate-400">
              Aroha is currently an early-stage project and the platform
              is actively evolving as we build and test its underlying
              technology.
            </p>

            <p className="mt-4 max-w-[700px] text-[15px] leading-7 text-slate-400">
              If you care about public health, community-driven technology,
              or building tools that can make disease surveillance more
              responsive, we'd love to hear from you.
            </p>

            <a
              href="https://github.com/Aroha-V/Aroha"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex mt-6 px-5 py-2.5 rounded-lg bg-[#5B6CF9] text-white text-sm font-medium hover:opacity-90 transition"
            >
              View the project on GitHub
            </a>
          </div>
        </section>
      </main>

      <Footer />
    </>
  );
}

function Step({ number, title, description }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-6">
      <span className="text-xs font-semibold text-[#5B6CF9]">
        {number}
      </span>

      <h3 className="mt-4 text-lg font-semibold">
        {title}
      </h3>

      <p className="mt-3 text-sm leading-6 text-slate-400">
        {description}
      </p>
    </div>
  );
}

function Feature({ title, description }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-6">
      <h3 className="text-lg font-semibold">
        {title}
      </h3>

      <p className="mt-3 text-sm leading-6 text-slate-400">
        {description}
      </p>
    </div>
  );
}