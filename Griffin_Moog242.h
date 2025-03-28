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
        // Tanh helper class with compiletime option for a polynomial approximation.
        template <bool UseApprox>
        struct TanhHelper {
            static inline float tanh(float x) { return std::tanh(x); }
        };

        // Specialization for fast polynomial tanh approximation.
        template <>
        struct TanhHelper<true> {
            static inline float tanh(float x) {
                float x2 = x * x;
                // Approximation
                float sh = x * (1.f + x2 * (1.f / 6.f + x2 * (1.f / 120.f)));
                return sh / std::sqrt(1.f + sh * sh);
            }
        };
    }

    // Monophonic ladder filter voice class.
    template <bool UseApproxTanh = USE_APPROX_TANH>
    class AudioEffect {
    public:
        AudioEffect(float initCutoff = 5000.f, float initResonance = 1.f)
            : cutoff(initCutoff), resonance(initResonance), keytrackAmount(0.f), note(60),
            fbk(1.f), b(0.65f), errorThresh(0.000001f)
        {
            for (int i = 0; i < 5; i++)
                y[i] = 0.f;
            s1 = s2 = s3 = s4 = s5 = 0.f;
        }

        // Initialize with the current sample rate.
        void prepare(double sampleRate) {
            sr = sampleRate;
        }

        // Reset internal state.
        void reset() {
            for (int i = 0; i < 5; i++)
                y[i] = 0.f;
            s1 = s2 = s3 = s4 = s5 = 0.f;
        }

        // Process a single sample.
        inline float processSample(float x) {
            float effectiveCutoff = cutoff * std::exp2((float(note) - 60.f) * keytrackAmount / 12.f);
            effectiveCutoff = std::max(20.f, std::min(effectiveCutoff, float(sr * 0.49f)));
            float g = std::tan(effectiveCutoff * M_PI / sr);
            const int maxIter = 50;
            int iter = 0;
            float residue = 1e6f;
            float Fc_hp = 8.f;
            float g_hp = Fc_hp * M_PI / sr;
            float g_den = 2.f * g_hp + 1.f;

            while (std::abs(residue) > errorThresh && iter < maxIter) {
                // Save only previous y[3] for convergence test.
                float old_y3 = y[3];

                float tanh_x = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh(x - resonance * y[3] + y[4]);
                float tanh_y0 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh(y[0]);
                float tanh_y1 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh(y[1]);
                float tanh_y2 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh(y[2]);
                float tanh_y3 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh(y[3]);
                float tanh_y5 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh(fbk * (y[3] - b));

                float F0 = g * (tanh_x - tanh_y0) + s1 - y[0];
                float F1 = g * (tanh_y0 - tanh_y1) + s2 - y[1];
                float F2 = g * (tanh_y1 - tanh_y2) + s3 - y[2];
                float F3 = g * (tanh_y2 - tanh_y3) + s4 - y[3];
                float F4 = (tanh_y5 + s5) / g_den - y[4];

                float help_x = 1.f - tanh_x * tanh_x;
                float help_y0 = 1.f - tanh_y0 * tanh_y0;
                float help_y1 = 1.f - tanh_y1 * tanh_y1;
                float help_y2 = 1.f - tanh_y2 * tanh_y2;
                float help_y3 = 1.f - tanh_y3 * tanh_y3;
                float help_y5 = 1.f - tanh_y5 * tanh_y5;
                float minus_g = -g;

                // Jacobian elements.
                float j00 = minus_g * help_y0 - 1.f;
                float j03 = minus_g * resonance * help_x;
                float j04 = g * help_x;
                float j10 = g * help_y0;
                float j11 = minus_g * help_y1 - 1.f;
                float j21 = g * help_y1;
                float j22 = minus_g * help_y2 - 1.f;
                float j32 = g * help_y2;
                float j33 = minus_g * help_y3 - 1.f;
                float j43 = fbk * help_y5 / g_den;

                float temp = j10 * j21 * j32;
                float den = j00 * j11 * j22 * j33 - j03 * temp - j04 * temp * j43;

                float T1 = j11 * j22 * j33;
                float T2 = j03 * j21 * j32;
                float T3 = j04 * j21 * j32 * j43;
                float T4 = j03 * j11 * j32;
                float T5 = j04 * j11 * j32 * j43;
                float T6 = j03 * j11 * j22;
                float T7 = j04 * j11 * j22 * j43;
                float T8 = j04 * j11 * j22 * j33;
                float delta0 = (F0 * T1 - F1 * (T2 + T3) + F2 * (T4 + T5) - F3 * (T6 + T7) + F4 * T8) / den;

                float T9 = j10 * j22 * j33;
                float T10 = j00 * j22 * j33;
                float T11 = j03 * j10 * j32;
                float T12 = j04 * j10 * j32 * j43;
                float T13 = j03 * j10 * j22;
                float T14 = j04 * j10 * j22 * j43;
                float T15 = j04 * j10 * j22 * j33;
                float delta1 = (F0 * T9 - F1 * T10 + F2 * (T11 + T12) - F3 * (T13 + T14) + F4 * T15) / den;

                float T16 = j10 * j21 * j33;
                float T17 = j00 * j21 * j33;
                float T18 = j00 * j11 * j33;
                float T19 = j03 * j10 * j21;
                float T20 = j04 * j10 * j21 * j43;
                float T21 = j04 * j10 * j21 * j33;
                float delta2 = (F0 * T16 - F1 * T17 + F2 * T18 - F3 * (T19 + T20) + F4 * T21) / den;

                float T22 = j10 * j21 * j32;
                float T23 = j00 * j21 * j32;
                float T24 = j00 * j11 * j32;
                float T25 = j00 * j11 * j22;
                float T26 = j04 * j10 * j21 * j32;
                float delta3 = (F0 * T22 - F1 * T23 + F2 * T24 - F3 * T25 + F4 * T26) / den;

                float T27 = j10 * j21 * j32 * j43;
                float T28 = j00 * j21 * j32 * j43;
                float T29 = j00 * j11 * j32 * j43;
                float T30 = j00 * j11 * j22 * j43;
                float T31 = j00 * j11 * j22 * j33;
                float T32 = j03 * j10 * j21 * j32;
                float delta4 = (F0 * T27 - F1 * T28 + F2 * T29 - F3 * T30 + F4 * (T31 - T32)) / den;

                y[0] -= delta0;
                y[1] += delta1;
                y[2] -= delta2;
                y[3] += delta3;
                y[4] += delta4;

                residue = y[3] - old_y3;
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

        float cutoffFrequency = 5000.f;
        float resonance = 1.f;
        float keytrackAmount = 0.f;

        SmoothedValue<float> cutoffSmooth;
        SmoothedValue<float> resonanceSmooth;
        SmoothedValue<float> keytrackSmooth;

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
                float cVal = cutoffSmooth.getNextValue();
                float rVal = resonanceSmooth.getNextValue();
                float ktVal = keytrackSmooth.getNextValue();
                float inL = leftChannelData[i];
                float inR = rightChannelData[i];
                float outL = 0.f;
                float outR = 0.f;

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
                for (auto& voice : filtersLeft)
                    outL += voice.processSample(inL);
                for (auto& voice : filtersRight)
                    outR += voice.processSample(inR);

                leftChannelData[i] = outL / NV;
                rightChannelData[i] = outR / NV;
            }
        }

        template <typename FrameDataType>
        void processFrame(FrameDataType& data) {}

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
                parameter::data p("Resonance", { 0.1, 5.0, 0.01 });
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

        void setExternalData(const ExternalData& ed, int index) {}

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
