#pragma once
#include <JuceHeader.h>
#include <cmath>
#include <array>
#include <algorithm>

// High Cpu usage, Good Sound

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef TOLERANCE_MULTIPLIER
#define TOLERANCE_MULTIPLIER 0.5f
#endif

#ifndef NOISE_FEEDBACK_VOLUME
#define NOISE_FEEDBACK_VOLUME 0.008f
#endif

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

    // JunoFilterStereoDual processes stereo samples using a Newton–Raphson solver.
    // Internal drive is fixed at 7.5; the nonlinearity is computed via a LUT.
    class JunoFilterStereoDual {
    public:
        JunoFilterStereoDual()
            : cutoff(1000.f), resonance(1.f), drive(7.5f), sr(44100.0),
            errorThresh(0.000001f)
        {
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
            // Precompute invariant values.
            float baseG = std::tan(cutoff * M_PI / sr);
            const float g0L = baseG * stageToleranceLeft[0];
            const float g1L = baseG * stageToleranceLeft[1];
            const float g2L = baseG * stageToleranceLeft[2];
            const float g3L = baseG * stageToleranceLeft[3];
            const float tol0L = stageToleranceLeft[0];
            const float tol1L = stageToleranceLeft[1];
            const float tol2L = stageToleranceLeft[2];
            const float tol3L = stageToleranceLeft[3];

            const float g0R = baseG * stageToleranceRight[0];
            const float g1R = baseG * stageToleranceRight[1];
            const float g2R = baseG * stageToleranceRight[2];
            const float g3R = baseG * stageToleranceRight[3];
            const float tol0R = stageToleranceRight[0];
            const float tol1R = stageToleranceRight[1];
            const float tol2R = stageToleranceRight[2];
            const float tol3R = stageToleranceRight[3];

            float noiseL = (NOISE_FEEDBACK_VOLUME > 0.0f)
                ? NOISE_FEEDBACK_VOLUME * (randGen.nextFloat() * 2.f - 1.f)
                : 0.f;
            float noiseR = (NOISE_FEEDBACK_VOLUME > 0.0f)
                ? NOISE_FEEDBACK_VOLUME * (randGen.nextFloat() * 2.f - 1.f)
                : 0.f;

            // --- Left channel processing ---
            float y0 = yL[0], y1 = yL[1], y2 = yL[2], y3 = yL[3];
            const float inValL = inL;
            for (int iter = 0; iter < 50; ++iter) {
                float prev_y3 = y3;
                float nl0 = transistorHelpers::advancedTransistorNonlinearity((inValL - y0 - resonance * y3 + noiseL) * tol0L, drive);
                float nl1 = transistorHelpers::advancedTransistorNonlinearity((y0 - y1) * tol1L, drive);
                float nl2 = transistorHelpers::advancedTransistorNonlinearity((y1 - y2) * tol2L, drive);
                float nl3 = transistorHelpers::advancedTransistorNonlinearity((y2 - y3) * tol3L, drive);

                float f0 = g0L * nl0 + sL1 - y0;
                float f1 = g1L * nl1 + sL2 - y1;
                float f2 = g2L * nl2 + sL3 - y2;
                float f3 = g3L * nl3 + sL4 - y3;

                float h0 = 1.f - nl0 * nl0;
                float h1 = 1.f - nl1 * nl1;
                float h2 = 1.f - nl2 * nl2;
                float h3 = 1.f - nl3 * nl3;

                float j00 = -g0L * h0 - 1.f;
                float j03 = -g0L * resonance * h0;
                float j10 = g1L * h1;
                float j11 = -g1L * h1 - 1.f;
                float j21 = g2L * h2;
                float j22 = -g2L * h2 - 1.f;
                float j32 = g3L * h3;
                float j33 = -g3L * h3 - 1.f;

                float den = j00 * j11 * j22 * j33 - j03 * j10 * j21 * j32;
                float delta0 = (f1 * j03 * j21 * j32 - f0 * j11 * j22 * j33 - f2 * j03 * j11 * j32 + f3 * j03 * j11 * j22) / den;
                float delta1 = (f0 * j10 * j22 * j33 - f1 * j00 * j22 * j33 + f2 * j03 * j10 * j32 - f3 * j03 * j10 * j22) / den;
                float delta2 = (f1 * j00 * j21 * j33 - f0 * j10 * j21 * j33 - f2 * j00 * j11 * j33 + f3 * j03 * j10 * j21) / den;
                float delta3 = (f0 * j10 * j21 * j32 - f1 * j00 * j21 * j32 + f2 * j00 * j11 * j32 - f3 * j00 * j11 * j22) / den;

                y0 += delta0;
                y1 += delta1;
                y2 += delta2;
                y3 += delta3;
                if (std::abs(y3 - prev_y3) <= errorThresh)
                    break;
            }
            sL1 = 2.f * y0 - sL1;
            sL2 = 2.f * y1 - sL2;
            sL3 = 2.f * y2 - sL3;
            sL4 = 2.f * y3 - sL4;
            float outL = y3;
            yL[0] = y0; yL[1] = y1; yL[2] = y2; yL[3] = y3;

            // --- Right channel processing ---
            float yr0 = yR[0], yr1 = yR[1], yr2 = yR[2], yr3 = yR[3];
            const float inValR = inR;
            for (int iter = 0; iter < 50; ++iter) {
                float prev_yr3 = yr3;
                float nlr0 = transistorHelpers::advancedTransistorNonlinearity((inValR - yr0 - resonance * yr3 + noiseR) * tol0R, drive);
                float nlr1 = transistorHelpers::advancedTransistorNonlinearity((yr0 - yr1) * tol1R, drive);
                float nlr2 = transistorHelpers::advancedTransistorNonlinearity((yr1 - yr2) * tol2R, drive);
                float nlr3 = transistorHelpers::advancedTransistorNonlinearity((yr2 - yr3) * tol3R, drive);

                float fr0 = g0R * nlr0 + sR1 - yr0;
                float fr1 = g1R * nlr1 + sR2 - yr1;
                float fr2 = g2R * nlr2 + sR3 - yr2;
                float fr3 = g3R * nlr3 + sR4 - yr3;

                float hr0 = 1.f - nlr0 * nlr0;
                float hr1 = 1.f - nlr1 * nlr1;
                float hr2 = 1.f - nlr2 * nlr2;
                float hr3 = 1.f - nlr3 * nlr3;

                float jr00 = -g0R * hr0 - 1.f;
                float jr03 = -g0R * resonance * hr0;
                float jr10 = g1R * hr1;
                float jr11 = -g1R * hr1 - 1.f;
                float jr21 = g2R * hr2;
                float jr22 = -g2R * hr2 - 1.f;
                float jr32 = g3R * hr3;
                float jr33 = -g3R * hr3 - 1.f;

                float denR = jr00 * jr11 * jr22 * jr33 - jr03 * jr10 * jr21 * jr32;
                float dR0 = (fr1 * jr03 * jr21 * jr32 - fr0 * jr11 * jr22 * jr33 - fr2 * jr03 * jr11 * jr32 + fr3 * jr03 * jr11 * jr22) / denR;
                float dR1 = (fr0 * jr10 * jr22 * jr33 - fr1 * jr00 * jr22 * jr33 + fr2 * jr03 * jr10 * jr32 - fr3 * jr03 * jr10 * jr22) / denR;
                float dR2 = (fr1 * jr00 * jr21 * jr33 - fr0 * jr10 * jr21 * jr33 - fr2 * jr00 * jr11 * jr33 + fr3 * jr03 * jr10 * jr21) / denR;
                float dR3 = (fr0 * jr10 * jr21 * jr32 - fr1 * jr00 * jr21 * jr32 + fr2 * jr00 * jr11 * jr32 - fr3 * jr00 * jr11 * jr22) / denR;

                yr0 += dR0;
                yr1 += dR1;
                yr2 += dR2;
                yr3 += dR3;
                if (std::abs(yr3 - prev_yr3) <= errorThresh)
                    break;
            }
            sR1 = 2.f * yr0 - sR1;
            sR2 = 2.f * yr1 - sR2;
            sR3 = 2.f * yr2 - sR3;
            sR4 = 2.f * yr3 - sR4;
            float outR = yr3;
            yR[0] = yr0; yR[1] = yr1; yR[2] = yr2; yR[3] = yr3;

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

        juce::SmoothedValue<float> cutoffSmooth;
        juce::SmoothedValue<float> resonanceSmooth;

        PolyData<JunoFilterStereoDual, NV> filters;

        void prepare(PrepareSpecs specs) {
            double sampleRate = specs.sampleRate;
            cutoffSmooth.reset(sampleRate, 0.01);
            resonanceSmooth.reset(sampleRate, 0.01);
            cutoffSmooth.setCurrentAndTargetValue(cutoffFrequency);
            resonanceSmooth.setCurrentAndTargetValue(resonance);

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

        template <typename ProcessDataType>
        void process(ProcessDataType& data) {
            auto& fixData = data.template as<ProcessData<getFixChannelAmount()>>();
            auto audioBlock = fixData.toAudioBlock();
            float* leftChannel = audioBlock.getChannelPointer(0);
            float* rightChannel = audioBlock.getChannelPointer(1);
            int numSamples = static_cast<int>(data.getNumSamples());

            for (int i = 0; i < numSamples; ++i) {
                float inL_orig = leftChannel[i];
                float inR_orig = rightChannel[i];
                float inL = std::tanh(1.5f * inL_orig) / std::tanh(1.5f);
                float inR = std::tanh(1.5f * inR_orig) / std::tanh(1.5f);

                float cVal = cutoffSmooth.getNextValue();
                float rVal = resonanceSmooth.getNextValue();
                for (auto& voice : filters) {
                    voice.setCutoff(cVal);
                    voice.setResonance(rVal);
                    voice.setDrive(7.5f);
                }

                float outL = 0.f;
                float outR = 0.f;
                for (auto& voice : filters) {
                    auto outs = voice.processSample(inL, inR);
                    outL += outs.first;
                    outR += outs.second;
                }
                outL /= NV;
                outR /= NV;

                leftChannel[i] = outL;
                rightChannel[i] = outR;
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
