#pragma once
#include <JuceHeader.h>
#include <cmath>

#ifndef USE_APPROX_TANH
#define USE_APPROX_TANH true
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace project {

    using namespace juce;
    using namespace hise;
    using namespace scriptnode;

    namespace FunctionsClasses {
        // Tanh helper class with a compile-time option for a polynomial approximation.
        template <bool UseApprox>
        struct TanhHelper {
            static inline float tanh(float x) { return std::tanh(x); }
        };

        // Specialization for fast polynomial tanh approximation.
        template <>
        struct TanhHelper<true> {
            static inline float tanh(float x) {
                float x2 = x * x;
                // Approximation: tanh(x) approx = sh/sqrt(1+sh^2), with sh = x*(1 + x^2/6 + x^4/120)
                float sh = x * (1.f + x2 * (1.f / 6.f + x2 * (1.f / 120.f)));
                return sh / std::sqrt(1.f + sh * sh);
            }
        };
    }

    // Monophonic ladder filter voice class.
    // Keytracking: effectiveCutoff = cutoff * exp2((note-60)*keytrackAmount/12)
    // No Eigen dependency; uses a manual Newton-Raphson update.
    template <bool UseApproxTanh = USE_APPROX_TANH>
    class AudioEffect {
    public:
        AudioEffect(float initCutoff = 5000.f, float initResonance = 1.f)
            : cutoff(initCutoff), resonance(initResonance), keytrackAmount(0.0f), note(60)
        {
            fbk = 1.f;  // feedback factor (constant)
            b = 0.65f;  // bias value
            errorThresh = 0.000001f;
            for (int i = 0; i < 5; i++)
                y[i] = 0.f;
            s1 = s2 = s3 = s4 = s5 = 0.f;
        }

        // Initialize with current sample rate.
        void prepare(double sampleRate) {
            sr = sampleRate;
        }

        // Reset internal state (used by polyphonic reset).
        void reset() {
            for (int i = 0; i < 5; i++)
                y[i] = 0.f;
            s1 = s2 = s3 = s4 = s5 = 0.f;
        }

        // Process a single sample.
        inline float processSample(float x) {
            // Compute effective cutoff with keytracking.
            float effectiveCutoff = cutoff * std::exp2(((float)note - 60.f) * keytrackAmount / 12.f);
            if (effectiveCutoff < 20.f)
                effectiveCutoff = 20.f;
            if (effectiveCutoff > sr * 0.49f)
                effectiveCutoff = sr * 0.49f;

            // Compute g from effective cutoff.
            float g = std::tan(effectiveCutoff * M_PI / sr);
            const int maxIter = 50;
            int iter = 0;
            float residue = 1e6f;
            float y_prev[5];
            float F[5];
            float j00, j03, j04, j10, j11, j21, j22, j32, j33, j43;
            float den = 0.f;
            // Pre-calculate highpass variables for F[4]
            float Fc_hp = 8.f;
            float g_hp = Fc_hp * M_PI / sr;
            float g_den = 2.f * g_hp + 1.f;

            // Newton-Raphson iteration.
            while (std::abs(residue) > errorThresh && iter < maxIter) {
                // Save previous state.
                for (int k = 0; k < 5; k++)
                    y_prev[k] = y[k];

                float tanh_x = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh(x - resonance * y[3] + y[4]);
                float tanh_y1 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh(y[0]);
                float tanh_y2 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh(y[1]);
                float tanh_y3 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh(y[2]);
                float tanh_y4 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh(y[3]);
                float tanh_y5 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh(fbk * (y[3] - b));

                F[0] = g * (tanh_x - tanh_y1) + s1 - y[0];
                F[1] = g * (tanh_y1 - tanh_y2) + s2 - y[1];
                F[2] = g * (tanh_y2 - tanh_y3) + s3 - y[2];
                F[3] = g * (tanh_y3 - tanh_y4) + s4 - y[3];
                F[4] = (tanh_y5 + s5) / g_den - y[4];

                float help_x = 1.f - (tanh_x * tanh_x);
                float help_y1 = 1.f - (tanh_y1 * tanh_y1);
                float help_y2 = 1.f - (tanh_y2 * tanh_y2);
                float help_y3 = 1.f - (tanh_y3 * tanh_y3);
                float help_y4 = 1.f - (tanh_y4 * tanh_y4);
                float help_y5 = 1.f - (tanh_y5 * tanh_y5);
                float minus_g = -g;

                // Jacobian nonzero elements.
                j00 = minus_g * help_y1 - 1.f;
                j03 = minus_g * resonance * help_x;
                j04 = g * help_x;
                j10 = g * help_y1;
                j11 = minus_g * help_y2 - 1.f;
                j21 = g * help_y2;
                j22 = minus_g * help_y3 - 1.f;
                j32 = g * help_y3;
                j33 = minus_g * help_y4 - 1.f;
                j43 = fbk * help_y5 / g_den;

                den = j00 * j11 * j22 * j33 - j03 * j10 * j21 * j32 - j04 * j10 * j21 * j32 * j43;

                // Manual Newton-Raphson update.
                y[0] = y[0] - (F[0] * j11 * j22 * j33 - F[1] * (j03 * j21 * j32 + j04 * j21 * j32 * j43) +
                    F[2] * (j03 * j11 * j32 + j04 * j11 * j32 * j43) - F[3] * (j03 * j11 * j22 + j04 * j11 * j22 * j43) +
                    F[4] * j04 * j11 * j22 * j33) / den;
                y[1] = y[1] + (F[0] * j10 * j22 * j33 - F[1] * j00 * j22 * j33 +
                    F[2] * (j03 * j10 * j32 + j04 * j10 * j32 * j43) - F[3] * (j03 * j10 * j22 + j04 * j10 * j22 * j43) +
                    F[4] * j04 * j10 * j22 * j33) / den;
                y[2] = y[2] - (F[0] * j10 * j21 * j33 - F[1] * j00 * j21 * j33 +
                    F[2] * j00 * j11 * j33 - F[3] * (j03 * j10 * j21 + j04 * j10 * j21 * j43) +
                    F[4] * j04 * j10 * j21 * j33) / den;
                y[3] = y[3] + (F[0] * j10 * j21 * j32 - F[1] * j00 * j21 * j32 +
                    F[2] * j00 * j11 * j32 - F[3] * j00 * j11 * j22 +
                    F[4] * j04 * j10 * j21 * j32) / den;
                y[4] = y[4] + (F[0] * j10 * j21 * j32 * j43 - F[1] * j00 * j21 * j32 * j43 +
                    F[2] * j00 * j11 * j32 * j43 - F[3] * j00 * j11 * j22 * j43 +
                    F[4] * (j00 * j11 * j22 * j33 - j03 * j10 * j21 * j32)) / den;

                residue = y[3] - y_prev[3];
                iter++;
            }

            // Update capacitor states.
            s1 = 2.f * y[0] - s1;
            s2 = 2.f * y[1] - s2;
            s3 = 2.f * y[2] - s3;
            s4 = 2.f * y[3] - s4;
            s5 = 2.f * (y[4] - FunctionsClasses::TanhHelper<UseApproxTanh>::tanh(fbk * (y[3] - b))) - s5;

            // Output from the fourth stage.
            return y[3];
        }

        // Parameter setters.
        void setCutoff(float newCutoff) { cutoff = newCutoff; }
        void setResonance(float newResonance) { resonance = newResonance; }
        void setKeytrack(float newKeytrack) { keytrackAmount = newKeytrack; }
        void setNoteNumber(int newNote) { note = newNote; }

    private:
        double sr;
        float cutoff;
        float resonance;
        float keytrackAmount;
        int note;
        float fbk;
        float b;
        float y[5];
        float s1, s2, s3, s4, s5;
        float errorThresh;
    };

    // Polyphonic node implementing polyphony, per-sample smoothing, and keytracking.
    // Uses a PolyData container for left/right channels.
    template <int NV>
    struct Griffin_Moog242 : public data::base {
        SNEX_NODE(Griffin_Moog242);

        struct MetadataClass {
            SN_NODE_ID("Griffin_Moog242");
        };

        static constexpr bool isModNode() { return false; }
        static constexpr bool isPolyphonic() { return NV > 1; }
        static constexpr bool hasTail() { return false; }
        static constexpr bool isSuspendedOnSilence() { return false; }
        static constexpr int getFixChannelAmount() { return 2; }

        static constexpr int NumTables = 0;
        static constexpr int NumSliderPacks = 0;
        static constexpr int NumAudioFiles = 0;
        static constexpr int NumFilters = 0;
        static constexpr int NumDisplayBuffers = 0;

        // Outer-level parameters and their smoothed versions.
        float cutoffFrequency = 5000.0f;
        float resonance = 1.0f;
        float keytrackAmount = 0.0f;

        SmoothedValue<float> cutoffSmooth;
        SmoothedValue<float> resonanceSmooth;
        SmoothedValue<float> keytrackSmooth;

        // Polyphonic voices for left and right channels.
        PolyData<AudioEffect<>, NV> filtersLeft;
        PolyData<AudioEffect<>, NV> filtersRight;

        void prepare(PrepareSpecs specs) {
            double sampleRate = specs.sampleRate;
            cutoffSmooth.reset(sampleRate, 0.01);
            resonanceSmooth.reset(sampleRate, 0.01);
            keytrackSmooth.reset(sampleRate, 0.01);

            cutoffSmooth.setCurrentAndTargetValue(cutoffFrequency);
            resonanceSmooth.setCurrentAndTargetValue(resonance);
            keytrackSmooth.setCurrentAndTargetValue(keytrackAmount);

            filtersLeft.prepare(specs);
            filtersRight.prepare(specs);
            for (auto& voice : filtersLeft)
                voice.prepare(sampleRate);
            for (auto& voice : filtersRight)
                voice.prepare(sampleRate);
        }

        // Iterate over each voice to reset its state.
        void reset() {
            for (auto& voice : filtersLeft)
                voice.reset();
            for (auto& voice : filtersRight)
                voice.reset();
        }

        template <typename ProcessDataType>
        void process(ProcessDataType& data) {
            auto& fixData = data.template as<ProcessData<getFixChannelAmount()>>();
            auto audioBlock = fixData.toAudioBlock();
            float* leftChannelData = audioBlock.getChannelPointer(0);
            float* rightChannelData = audioBlock.getChannelPointer(1);
            int numSamples = static_cast<int>(data.getNumSamples());

            for (int i = 0; i < numSamples; ++i) {
                // Get per-sample smoothed parameters.
                float cVal = cutoffSmooth.getNextValue();
                float rVal = resonanceSmooth.getNextValue();
                float ktVal = keytrackSmooth.getNextValue();

                // Update all voices.
                for (auto& voice : filtersLeft) {
                    voice.setCutoff(cVal);
                    voice.setResonance(rVal);
                    voice.setKeytrack(ktVal);
                }
                for (auto& voice : filtersRight) {
                    voice.setCutoff(cVal);
                    voice.setResonance(rVal);
                    voice.setKeytrack(ktVal);
                }

                float inL = leftChannelData[i];
                float inR = rightChannelData[i];

                // Process sample for each voice and sum outputs.
                float outL = 0.0f;
                float outR = 0.0f;
                for (auto& voice : filtersLeft)
                    outL += voice.processSample(inL);
                for (auto& voice : filtersRight)
                    outR += voice.processSample(inR);

                // Average output over voices.
                leftChannelData[i] = outL / NV;
                rightChannelData[i] = outR / NV;
            }
        }

        template <typename FrameDataType>
        void processFrame(FrameDataType& data) {}

        // Parameter handling.
        template <int P>
        void setParameter(double v) {
            if (P == 0) {
                cutoffFrequency = static_cast<float>(v);
                cutoffSmooth.setTargetValue(cutoffFrequency);
            }
            else if (P == 1) {
                resonance = static_cast<float>(v);
                resonanceSmooth.setTargetValue(resonance);
            }
            else if (P == 2) {
                keytrackAmount = static_cast<float>(v);
                keytrackSmooth.setTargetValue(keytrackAmount);
            }
        }

        void createParameters(ParameterDataList& data) {
            {
                parameter::data p("Cutoff", { 20.0, 20000.0, 0.01 });
                registerCallback<0>(p);
                p.setDefaultValue(5000.0);
                data.add(std::move(p));
            }
            {
                parameter::data p("Resonance", { 0.1, 10.0, 0.01 });
                registerCallback<1>(p);
                p.setDefaultValue(1.0);
                data.add(std::move(p));
            }
            {
                parameter::data p("Keytrack", { -1.0, 1.0, 0.01 });
                registerCallback<2>(p);
                p.setDefaultValue(0.0);
                data.add(std::move(p));
            }
        }

        void setExternalData(const ExternalData& ed, int index) {
        }

        void handleHiseEvent(HiseEvent& e) {
            if (e.isNoteOn()) {
                for (auto& voice : filtersLeft)
                    voice.setNoteNumber(e.getNoteNumber());
                for (auto& voice : filtersRight)
                    voice.setNoteNumber(e.getNoteNumber());
            }
        }
    };

} // namespace project
