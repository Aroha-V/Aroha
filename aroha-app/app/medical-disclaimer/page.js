import Navbar from '@/components/Navbar/Navbar';
import Footer from '@/components/Footer/Footer';

export default function MedicalDisclaimerPage() {
  return (
    <>
      <Navbar />

      <main className="min-h-screen">
        <section className="max-w-[900px] mx-auto px-6 pt-24 pb-24">
          <div className="max-w-[760px]">
            <p className="text-sm font-medium text-[#5B6CF9] mb-4">
              MEDICAL DISCLAIMER
            </p>

            <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
              Information, not medical advice.
            </h1>

            <p className="mt-6 text-[15px] leading-7 text-slate-400">
              Aroha is an informational disease surveillance and health
              information platform.
            </p>
          </div>

          <div className="mt-14 space-y-10">

            <section>
              <h2 className="text-xl font-semibold">
                Not a medical diagnosis
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Information provided by Aroha, including AI-generated
                responses, symptom guidance, disease trends and risk
                information, is not a medical diagnosis and is not a
                substitute for professional medical advice, diagnosis or
                treatment.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                Consult a healthcare professional
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Always consult a qualified healthcare professional for
                medical concerns, symptoms or emergencies. Do not rely on
                information provided by Aroha as the sole basis for making
                medical decisions.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold">
                Disease surveillance information
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Disease surveillance information is compiled from publicly
                available sources, including government and public-health
                reports.
              </p>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Data may be delayed, incomplete or subject to revision.
                Aroha does not represent or endorse any government agency
                unless explicitly stated.
              </p>
            </section>

            <section className="rounded-2xl border border-[#5B6CF9]/20 bg-[#5B6CF9]/[0.05] p-6">
              <h2 className="text-lg font-semibold">
                If this is an emergency
              </h2>

              <p className="mt-3 text-[15px] leading-7 text-slate-400">
                Do not use Aroha for emergency medical care. Contact
                appropriate emergency medical services or seek immediate
                help from a qualified healthcare professional.
              </p>
            </section>

          </div>
        </section>
      </main>

      <Footer />
    </>
  );
}