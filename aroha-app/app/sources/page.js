import Navbar from '@/components/Navbar/Navbar';
import Footer from '@/components/Footer/Footer';

export default function SourcesPage() {
  return (
    <>
      <Navbar />

      <main className="min-h-screen">
        <section className="max-w-[900px] mx-auto px-6 pt-24 pb-24">
          <div className="max-w-[760px]">
            <p className="text-sm font-medium text-[#5B6CF9] mb-4">
              DATA SOURCES
            </p>

            <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
              Where Aroha's data comes from.
            </h1>

            <p className="mt-6 text-[15px] leading-7 text-slate-400">
              Aroha uses disease surveillance information from publicly
              available sources to provide disease-related information
              and surveillance insights.
            </p>
          </div>

          <div className="mt-14 space-y-10">

            <section>
              <h2 className="text-xl font-semibold">
                Integrated Disease Surveillance Programme
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                The disease surveillance data used by Aroha is based on
                information associated with the Integrated Disease
                Surveillance Programme (IDSP), under the Ministry of Health
                & Family Welfare, Government of India.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                Data limitations
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Surveillance information may be delayed, incomplete or
                subject to revision. Aroha presents this information for
                informational and surveillance-related purposes and does
                not guarantee that every record represents the most
                up-to-date situation.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                Government source
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                For official information about India's disease surveillance
                programme, please refer to the official government source.
              </p>

              <a
                href="https://idsp.mohfw.gov.in"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex mt-5 px-5 py-2.5 rounded-lg bg-[#5B6CF9] text-white text-sm font-medium hover:opacity-90 transition"
              >
                Visit IDSP
              </a>
            </section>

            <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <h2 className="text-lg font-semibold">
                Source disclaimer
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Disease surveillance information is compiled from publicly
                available sources, including government and public-health
                reports. Data may be delayed, incomplete or subject to
                revision. Aroha does not represent or endorse any
                government agency unless explicitly stated.
              </p>
            </section>

          </div>
        </section>
      </main>

      <Footer />
    </>
  );
}