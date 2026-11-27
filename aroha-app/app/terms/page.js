import Navbar from '@/components/Navbar/Navbar';
import Footer from '@/components/Footer/Footer';

export default function TermsPage() {
  return (
    <>
      <Navbar />

      <main className="min-h-screen">
        <section className="max-w-[900px] mx-auto px-6 pt-24 pb-24">
          <div className="max-w-[760px]">
            <p className="text-sm font-medium text-[#5B6CF9] mb-4">
              TERMS OF USE
            </p>

            <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
              Using Aroha responsibly.
            </h1>

            <p className="mt-6 text-[15px] leading-7 text-slate-400">
              These terms describe the basic conditions for using the
              Aroha platform.
            </p>

            <p className="mt-3 text-sm text-slate-500">
              Last updated: September 2026
            </p>
          </div>

          <div className="mt-14 space-y-10">

            <section>
              <h2 className="text-xl font-semibold">
                1. About Aroha
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Aroha is an informational disease surveillance and health
                information platform. The platform is intended to help
                users explore disease-related information and surveillance
                patterns.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                2. Informational use
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Information provided through Aroha is for informational
                purposes. It should not be treated as medical advice,
                diagnosis or treatment.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                3. Accuracy of information
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Disease surveillance information may be delayed, incomplete
                or subject to revision. Aroha does not guarantee that all
                information presented through the platform is complete,
                current or error-free.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                4. AI-generated information
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Some responses provided by Aroha may be generated or
                assisted by artificial intelligence. AI-generated responses
                may contain errors and should be interpreted accordingly.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                5. Responsible use
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Users should not use Aroha to make decisions that require
                professional medical judgment. The platform should not be
                used as a replacement for qualified healthcare or public
                health professionals.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                6. Account security
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                If you create an account, you are responsible for keeping
                your login information secure and for activity carried out
                through your account.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                7. Changes to Aroha
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Aroha is an early-stage project. Features, information
                sources and functionality may change as the platform
                develops.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                8. Changes to these terms
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                These terms may be updated as the project develops.
                Updated terms will be published on this page.
              </p>
            </section>

          </div>
        </section>
      </main>

      <Footer />
    </>
  );
}