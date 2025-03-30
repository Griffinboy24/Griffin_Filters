#pragma once
#include <JuceHeader.h>
#include <cmath>
#include <array>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef TOLERANCE_MULTIPLIER
#define TOLERANCE_MULTIPLIER 0.5f
#endif

#ifndef NOISE_FEEDBACK_VOLUME
#define NOISE_FEEDBACK_VOLUME 0.008f
#endif

#ifndef INPUT_SIGNAL_THRESHOLD
#define INPUT_SIGNAL_THRESHOLD 1e-6f
#endif

// Fast polynomial tanh approximation.
template <bool UseApprox = false>
struct TanhHelper {
    static inline float tanh(float x) { return std::tanh(x); }
};

// Specialization for fast polynomial tanh approximation.
template <>
struct TanhHelper<true> {
    static inline float tanh(float x) {
        float x2 = x * x;
        float sh = x * (1.f + x2 * (1.f / 6.f + x2 * (1.f / 120.f)));
        return sh / std::sqrt(1.f + sh * sh);
    }
};

// Advanced transistor nonlinearity helper using a lookup table.
// Approximates f(x) = (2/M_PI)*atan(7.5*x) over [-maxInput, maxInput].
namespace transistorHelpers {

    inline float lookupAdvancedTransistorNonlinearity(float x) {
        static const int TABLE_SIZE = 4096;
        static const float maxInput = 2.0f;
        static const float invStep = (TABLE_SIZE - 1) / (2.0f * maxInput);
        static float lookupTable[TABLE_SIZE];
        static bool tableInitialized = false;
        if (!tableInitialized)
        {
            const float step = 2.0f * maxInput / (TABLE_SIZE - 1);
            for (int i = 0; i < TABLE_SIZE; i++)
            {
                float xi = -maxInput + i * step;
                lookupTable[i] = (2.0f / M_PI) * std::atan(7.5f * xi);
            }
            tableInitialized = true;
        }
        float clampedX = std::clamp(x, -maxInput, maxInput);
        float index = (clampedX + maxInput) * invStep;
        int indexInt = static_cast<int>(index);
        float frac = index - indexInt;
        return lookupTable[indexInt] * (1.0f - frac) + lookupTable[indexInt + 1] * frac;
    }

    // 'drive' is ignored since it is fixed to 7.5.
    inline float advancedTransistorNonlinearity(float x, float /*drive*/) {
        return lookupAdvancedTransistorNonlinearity(x);
    }
}

namespace project {

    using namespace juce;
    using namespace hise;
    using namespace scriptnode;

    // JunoFilterStereoDual processes stereo samples using a Newton-Raphson solver.
    // Internal drive is fixed at 7.5; the nonlinearity is computed via a lookup table.
    class JunoFilterStereoDual {
    public:
        JunoFilterStereoDual()
            : cutoff(1000.f), resonance(1.f), drive(7.5f), sr(44100.0),
            errorThresh(0.000001f)
        {
            // Candidate stage tolerance multipliers.
            static const std::array<std::array<float, 4>, 4> candidateLeft{ {
                {1.00f, 1.015f, 1.03f, 1.015f},
                {1.015f, 1.03f, 1.00f, 1.015f},
                {1.03f, 1.00f, 1.015f, 1.03f},
                {1.015f, 1.03f, 1.015f, 1.00f}
            } };
            static const std::array<std::array<float, 4>, 4> candidateRight{ {
                {1.03f, 1.015f, 1.00f, 1.015f},
                {1.015f, 1.00f, 1.03f, 1.015f},
                {1.00f, 1.015f, 1.03f, 1.00f},
                {1.015f, 1.03f, 1.00f, 1.015f}
            } };

            static int voiceCounter = 0;
            int myIndex = voiceCounter++;
            int leftConfig = myIndex % candidateLeft.size();
            int rightConfig = (myIndex + 1) % candidateRight.size();

            for (int i = 0; i < 4; ++i) {
                stageToleranceLeft[i] = 1.f + (candidateLeft[leftConfig][i] - 1.f) * TOLERANCE_MULTIPLIER;
                stageToleranceRight[i] = 1.f + (candidateRight[rightConfig][i] - 1.f) * TOLERANCE_MULTIPLIER;
            }
            for (int i = 0; i < 4; ++i) {
                yL[i] = yL_est[i] = FL[i] = 0.f;
                yR[i] = yR_est[i] = FR[i] = 0.f;
            }
            sL1 = sL2 = sL3 = sL4 = 0.f;
            sR1 = sR2 = sR3 = sR4 = 0.f;
        }

        void setCutoff(float newCutoff) { cutoff = newCutoff; }
        void setResonance(float newResonance) { resonance = newResonance; }
        void setDrive(float /*unused*/) { drive = 7.5f; }

        void prepare(double sampleRate) { sr = sampleRate; }

        void reset() {
            for (int i = 0; i < 4; ++i) {
                yL[i] = 0.f;
                yR[i] = 0.f;
            }
            sL1 = sL2 = sL3 = sL4 = 0.f;
            sR1 = sR2 = sR3 = sR4 = 0.f;
        }

        // Process one stereo sample.
        std::pair<float, float> processSample(float inL, float inR) {
            // Precompute filter coefficient based on cutoff.
            const float baseG = std::tan(cutoff * M_PI / sr);
            const float gLeft[4] = {
                baseG * stageToleranceLeft[0],
                baseG * stageToleranceLeft[1],
                baseG * stageToleranceLeft[2],
                baseG * stageToleranceLeft[3]
            };
            const float gRight[4] = {
                baseG * stageToleranceRight[0],
                baseG * stageToleranceRight[1],
                baseG * stageToleranceRight[2],
                baseG * stageToleranceRight[3]
            };

            // Compute noise once per channel only if there is an input signal.
            const float noiseL = (NOISE_FEEDBACK_VOLUME > 0.0f && std::abs(inL) > INPUT_SIGNAL_THRESHOLD)
                ? NOISE_FEEDBACK_VOLUME * (randGen.nextFloat() * 2.f - 1.f)
                : 0.f;
            const float noiseR = (NOISE_FEEDBACK_VOLUME > 0.0f && std::abs(inR) > INPUT_SIGNAL_THRESHOLD)
                ? NOISE_FEEDBACK_VOLUME * (randGen.nextFloat() * 2.f - 1.f)
                : 0.f;

            // Lambda to perform Newton-Raphson iteration for one channel.
            auto processChannel = [=](const float inVal, const float noise, float y[4], float s[4], const float g[4], const float tol[4]) -> void {
                for (int iter = 0; iter < 50; ++iter) {
                    const float prev_y3 = y[3];
                    const float nl0 = transistorHelpers::advancedTransistorNonlinearity((inVal - y[0] - resonance * y[3] + noise) * tol[0], drive);
                    const float nl1 = transistorHelpers::advancedTransistorNonlinearity((y[0] - y[1]) * tol[1], drive);
                    const float nl2 = transistorHelpers::advancedTransistorNonlinearity((y[1] - y[2]) * tol[2], drive);
                    const float nl3 = transistorHelpers::advancedTransistorNonlinearity((y[2] - y[3]) * tol[3], drive);
                    const float f0 = g[0] * nl0 + s[0] - y[0];
                    const float f1 = g[1] * nl1 + s[1] - y[1];
                    const float f2 = g[2] * nl2 + s[2] - y[2];
                    const float f3 = g[3] * nl3 + s[3] - y[3];
                    const float h0 = 1.f - nl0 * nl0;
                    const float h1 = 1.f - nl1 * nl1;
                    const float h2 = 1.f - nl2 * nl2;
                    const float h3 = 1.f - nl3 * nl3;
                    const float j00 = -g[0] * h0 - 1.f;
                    const float j03 = -g[0] * resonance * h0;
                    const float j10 = g[1] * h1;
                    const float j11 = -g[1] * h1 - 1.f;
                    const float j21 = g[2] * h2;
                    const float j22 = -g[2] * h2 - 1.f;
                    const float j32 = g[3] * h3;
                    const float j33 = -g[3] * h3 - 1.f;
                    const float den = j00 * j11 * j22 * j33 - j03 * j10 * j21 * j32;
                    const float delta0 = (f1 * j03 * j21 * j32 - f0 * j11 * j22 * j33 - f2 * j03 * j11 * j32 + f3 * j03 * j11 * j22) / den;
                    const float delta1 = (f0 * j10 * j22 * j33 - f1 * j00 * j22 * j33 + f2 * j03 * j10 * j32 - f3 * j03 * j10 * j22) / den;
                    const float delta2 = (f1 * j00 * j21 * j33 - f0 * j10 * j21 * j33 - f2 * j00 * j11 * j33 + f3 * j03 * j10 * j21) / den;
                    const float delta3 = (f0 * j10 * j21 * j32 - f1 * j00 * j21 * j32 + f2 * j00 * j11 * j32 - f3 * j00 * j11 * j22) / den;
                    y[0] += delta0;
                    y[1] += delta1;
                    y[2] += delta2;
                    y[3] += delta3;
                    if (std::abs(y[3] - prev_y3) <= errorThresh)
                        break;
                }
                s[0] = 2.f * y[0] - s[0];
                s[1] = 2.f * y[1] - s[1];
                s[2] = 2.f * y[2] - s[2];
                s[3] = 2.f * y[3] - s[3];
                };

            // Process left channel.
            float yLeft[4] = { yL[0], yL[1], yL[2], yL[3] };
            float sLeft[4] = { sL1, sL2, sL3, sL4 };
            processChannel(inL, noiseL, yLeft, sLeft, gLeft, stageToleranceLeft);
            yL[0] = yLeft[0]; yL[1] = yLeft[1]; yL[2] = yLeft[2]; yL[3] = yLeft[3];
            sL1 = sLeft[0]; sL2 = sLeft[1]; sL3 = sLeft[2]; sL4 = sLeft[3];
            const float outL = yLeft[3];

            // Process right channel.
            float yRight[4] = { yR[0], yR[1], yR[2], yR[3] };
            float sRight[4] = { sR1, sR2, sR3, sR4 };
            processChannel(inR, noiseR, yRight, sRight, gRight, stageToleranceRight);
            yR[0] = yRight[0]; yR[1] = yRight[1]; yR[2] = yRight[2]; yR[3] = yRight[3];
            sR1 = sRight[0]; sR2 = sRight[1]; sR3 = sRight[2]; sR4 = sRight[3];
            const float outR = yRight[3];

            return { outL, outR };
        }

    private:
        double sr;
        float cutoff;
        float resonance;
        float drive;
        float errorThresh;

        float yL[4], yL_est[4], FL[4];
        float sL1, sL2, sL3, sL4;
        float stageToleranceLeft[4];

        float yR[4], yR_est[4], FR[4];
        float sR1, sR2, sR3, sR4;
        float stageToleranceRight[4];

        juce::Random randGen;
    };

    // Polyphonic wrapper node.
    template <int NV>
    struct Griffin_Juno242 : public data::base {
        SNEX_NODE(Griffin_Juno242);

        struct MetadataClass {
            SN_NODE_ID("Griffin_Juno242");
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

        float cutoffFrequency = 1000.f;
        float resonance = 1.f;

        PolyData<JunoFilterStereoDual, NV> filters;

        // Prepare voices and update parameters.
        void prepare(PrepareSpecs specs) {
            const double sampleRate = specs.sampleRate;
            filters.prepare(specs);
            for (auto& voice : filters) {
                voice.prepare(sampleRate);
                voice.setCutoff(cutoffFrequency);
                voice.setResonance(resonance);
                voice.setDrive(7.5f);
            }
        }

        void reset() {
            for (auto& voice : filters)
                voice.reset();
        }

        // Process audio block.
        template <typename ProcessDataType>
        inline void process(ProcessDataType& data) {
            auto& fixData = data.template as<ProcessData<getFixChannelAmount()>>();
            auto audioBlock = fixData.toAudioBlock();
            float* leftChannel = audioBlock.getChannelPointer(0);
            float* rightChannel = audioBlock.getChannelPointer(1);
            const int numSamples = static_cast<int>(data.getNumSamples());

            // Precompute constant tanh value.
            const float tanhConst = TanhHelper<true>::tanh(1.5f);

            // Update voice parameters once per block.
            for (auto& voice : filters) {
                voice.setCutoff(cutoffFrequency);
                voice.setResonance(resonance);
                voice.setDrive(7.5f);
            }

            for (int i = 0; i < numSamples; ++i) {
                // Compute input scaling using the fast tanh approximation.
                const float inL = TanhHelper<true>::tanh(1.5f * leftChannel[i]) / tanhConst;
                const float inR = TanhHelper<true>::tanh(1.5f * rightChannel[i]) / tanhConst;

                float outL = 0.f, outR = 0.f;
                // Sum outputs from all voices.
                for (auto& voice : filters) {
                    auto outs = voice.processSample(inL, inR);
                    outL += outs.first;
                    outR += outs.second;
                }
                // Average across voices.
                leftChannel[i] = outL / NV;
                rightChannel[i] = outR / NV;
            }
        }

        template <typename FrameDataType>
        void processFrame(FrameDataType& data) {}

        template <int P>
        void setParameter(double v) {
            if (P == 0)
                cutoffFrequency = static_cast<float>(v);
            else if (P == 1)
                resonance = static_cast<float>(v);
        }

        void createParameters(ParameterDataList& data) {
            {
                parameter::data p("Cutoff", { 20.0, 4000.0, 0.00001 });
                registerCallback<0>(p);
                p.setDefaultValue(1000.0);
                data.add(std::move(p));
            }
            {
                parameter::data p("Resonance", { 0.1, 4.3, 0.00001 });
                registerCallback<1>(p);
                p.setDefaultValue(1.0);
                data.add(std::move(p));
            }
        }

        void setExternalData(const ExternalData& ed, int index) {}

        void handleHiseEvent(HiseEvent& e) {}
    };

} // namespace project
