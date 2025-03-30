#pragma once
#include <JuceHeader.h>
#include <cmath>
#include <array>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef NOISE_FEEDBACK_VOLUME
#define NOISE_FEEDBACK_VOLUME 0.008f
#endif

#ifndef INPUT_SIGNAL_THRESHOLD
#define INPUT_SIGNAL_THRESHOLD 1e-6f
#endif

// Fast tanh approximation
struct TanhHelper {
    static inline float tanh(float x) {
        float x2 = x * x;
        float sh = x * (1.f + x2 * (1.f / 6.f + x2 * (1.f / 120.f)));
        return sh / std::sqrt(1.f + sh * sh);
    }
};

// Transistor nonlinearity lookup table for (2/M_PI)*atan(7.5*x)
namespace transistorHelpers {
    inline float lookupAdvancedTransistorNonlinearity(float x) {
        static const int TABLE_SIZE = 4096;
        static const float maxInput = 2.0f;
        static const float invStep = (TABLE_SIZE - 1) / (2.0f * maxInput);
        static float lookupTable[TABLE_SIZE];
        static bool tableInitialized = false;
        if (!tableInitialized) {
            const float step = 2.0f * maxInput / (TABLE_SIZE - 1);
            for (int i = 0; i < TABLE_SIZE; i++) {
                float xi = -maxInput + i * step;
                lookupTable[i] = (2.0f / M_PI) * std::atan(7.5f * xi);
            }
            tableInitialized = true;
        }
        float clampedX = std::clamp(x, -maxInput, maxInput);
        float index = (clampedX + maxInput) * invStep;
        int indexInt = static_cast<int>(index);
        float frac = index - indexInt;
        return lookupTable[indexInt] * (1.f - frac) + lookupTable[indexInt + 1] * frac;
    }
    inline float advancedTransistorNonlinearity(float x, float /*drive*/) {
        return lookupAdvancedTransistorNonlinearity(x);
    }
}

namespace project {
    using namespace juce;
    using namespace hise;
    using namespace scriptnode;

    // Stereo filter using Newton-Raphson iteration.
    // Caches the filter coefficient (cachedBaseG) and maintains separate states for left/right channels.
    class JunoFilterStereo {
    public:
        JunoFilterStereo()
            : cutoff(1000.f), resonance(1.f), drive(7.5f), sr(44100.0),
            errorThresh(0.000001f), cachedBaseG(std::tan(1000.f * M_PI / 44100.0))
        {
            for (int i = 0; i < 4; ++i) {
                yL[i] = 0.f;
                yR[i] = 0.f;
            }
            sL[0] = sL[1] = sL[2] = sL[3] = 0.f;
            sR[0] = sR[1] = sR[2] = sR[3] = 0.f;
        }
        inline void setCutoff(float c) {
            if (cutoff != c) {
                cutoff = c;
                cachedBaseG = std::tan(cutoff * M_PI / sr);
            }
        }
        inline void setResonance(float r) { resonance = r; }
        inline void setDrive(float) { drive = 7.5f; }
        inline void prepare(double sr_) {
            sr = sr_;
            cachedBaseG = std::tan(cutoff * M_PI / sr);
        }
        inline void reset() {
            for (int i = 0; i < 4; ++i) {
                yL[i] = 0.f;
                yR[i] = 0.f;
            }
            sL[0] = sL[1] = sL[2] = sL[3] = 0.f;
            sR[0] = sR[1] = sR[2] = sR[3] = 0.f;
        }
        // Process one stereo sample; returns {left, right}.
        inline std::pair<float, float> processSample(float inL, float inR) {
            const float g = cachedBaseG;
            // Left channel
            const float noiseL = (NOISE_FEEDBACK_VOLUME > 0.f && std::abs(inL) > INPUT_SIGNAL_THRESHOLD) ?
                NOISE_FEEDBACK_VOLUME * (randGen.nextFloat() * 2.f - 1.f) : 0.f;
            for (int iter = 0; iter < 20; ++iter) {
                const float prev_yL3 = yL[3];
                float nl0 = transistorHelpers::advancedTransistorNonlinearity(inL - yL[0] - resonance * yL[3] + noiseL, drive);
                float nl1 = transistorHelpers::advancedTransistorNonlinearity(yL[0] - yL[1], drive);
                float nl2 = transistorHelpers::advancedTransistorNonlinearity(yL[1] - yL[2], drive);
                float nl3 = transistorHelpers::advancedTransistorNonlinearity(yL[2] - yL[3], drive);
                float f0 = g * nl0 + sL[0] - yL[0];
                float f1 = g * nl1 + sL[1] - yL[1];
                float f2 = g * nl2 + sL[2] - yL[2];
                float f3 = g * nl3 + sL[3] - yL[3];
                float h0 = 1.f - nl0 * nl0;
                float h1 = 1.f - nl1 * nl1;
                float h2 = 1.f - nl2 * nl2;
                float h3 = 1.f - nl3 * nl3;
                float j00 = -g * h0 - 1.f;
                float j03 = -g * resonance * h0;
                float j10 = g * h1;
                float j11 = -g * h1 - 1.f;
                float j21 = g * h2;
                float j22 = -g * h2 - 1.f;
                float j32 = g * h3;
                float j33 = -g * h3 - 1.f;
                float den = j00 * j11 * j22 * j33 - j03 * j10 * j21 * j32;
                yL[0] += (f1 * j03 * j21 * j32 - f0 * j11 * j22 * j33 - f2 * j03 * j11 * j32 + f3 * j03 * j11 * j22) / den;
                yL[1] += (f0 * j10 * j22 * j33 - f1 * j00 * j22 * j33 + f2 * j03 * j10 * j32 - f3 * j03 * j10 * j22) / den;
                yL[2] += (f1 * j00 * j21 * j33 - f0 * j10 * j21 * j33 - f2 * j00 * j11 * j33 + f3 * j03 * j10 * j21) / den;
                yL[3] += (f0 * j10 * j21 * j32 - f1 * j00 * j21 * j32 + f2 * j00 * j11 * j32 - f3 * j00 * j11 * j22) / den;
                if (std::abs(yL[3] - prev_yL3) <= errorThresh)
                    break;
            }
            sL[0] = 2.f * yL[0] - sL[0];
            sL[1] = 2.f * yL[1] - sL[1];
            sL[2] = 2.f * yL[2] - sL[2];
            sL[3] = 2.f * yL[3] - sL[3];
            // Right channel
            const float noiseR = (NOISE_FEEDBACK_VOLUME > 0.f && std::abs(inR) > INPUT_SIGNAL_THRESHOLD) ?
                NOISE_FEEDBACK_VOLUME * (randGen.nextFloat() * 2.f - 1.f) : 0.f;
            for (int iter = 0; iter < 20; ++iter) {
                const float prev_yR3 = yR[3];
                float nl0 = transistorHelpers::advancedTransistorNonlinearity(inR - yR[0] - resonance * yR[3] + noiseR, drive);
                float nl1 = transistorHelpers::advancedTransistorNonlinearity(yR[0] - yR[1], drive);
                float nl2 = transistorHelpers::advancedTransistorNonlinearity(yR[1] - yR[2], drive);
                float nl3 = transistorHelpers::advancedTransistorNonlinearity(yR[2] - yR[3], drive);
                float f0 = g * nl0 + sR[0] - yR[0];
                float f1 = g * nl1 + sR[1] - yR[1];
                float f2 = g * nl2 + sR[2] - yR[2];
                float f3 = g * nl3 + sR[3] - yR[3];
                float h0 = 1.f - nl0 * nl0;
                float h1 = 1.f - nl1 * nl1;
                float h2 = 1.f - nl2 * nl2;
                float h3 = 1.f - nl3 * nl3;
                float j00 = -g * h0 - 1.f;
                float j03 = -g * resonance * h0;
                float j10 = g * h1;
                float j11 = -g * h1 - 1.f;
                float j21 = g * h2;
                float j22 = -g * h2 - 1.f;
                float j32 = g * h3;
                float j33 = -g * h3 - 1.f;
                float den = j00 * j11 * j22 * j33 - j03 * j10 * j21 * j32;
                yR[0] += (f1 * j03 * j21 * j32 - f0 * j11 * j22 * j33 - f2 * j03 * j11 * j32 + f3 * j03 * j11 * j22) / den;
                yR[1] += (f0 * j10 * j22 * j33 - f1 * j00 * j22 * j33 + f2 * j03 * j10 * j32 - f3 * j03 * j10 * j22) / den;
                yR[2] += (f1 * j00 * j21 * j33 - f0 * j10 * j21 * j33 - f2 * j00 * j11 * j33 + f3 * j03 * j10 * j21) / den;
                yR[3] += (f0 * j10 * j21 * j32 - f1 * j00 * j21 * j32 + f2 * j00 * j11 * j32 - f3 * j00 * j11 * j22) / den;
                if (std::abs(yR[3] - prev_yR3) <= errorThresh)
                    break;
            }
            sR[0] = 2.f * yR[0] - sR[0];
            sR[1] = 2.f * yR[1] - sR[1];
            sR[2] = 2.f * yR[2] - sR[2];
            sR[3] = 2.f * yR[3] - sR[3];
            return { yL[3], yR[3] };
        }
    private:
        double sr;
        float cutoff, resonance, drive, errorThresh;
        float cachedBaseG;
        float yL[4], sL[4];
        float yR[4], sR[4];
        juce::Random randGen;
    };

    // Polyphonic stereo node.
    template <int NV>
    struct Griffin_Juno242 : public data::base {
        SNEX_NODE(Griffin_Juno242);
        struct MetadataClass { SN_NODE_ID("Griffin_Juno242"); };
        static constexpr bool isModNode() { return false; }
        static constexpr bool isPolyphonic() { return NV > 1; }
        static constexpr bool hasTail() { return false; }
        static constexpr bool isSuspendedOnSilence() { return false; }
        static constexpr int getFixChannelAmount() { return 2; }
        static constexpr int NumTables = 0, NumSliderPacks = 0, NumAudioFiles = 0, NumFilters = 0, NumDisplayBuffers = 0;

        float cutoffFrequency = 1000.f, resonance = 1.f;
        PolyData<JunoFilterStereo, NV> filters;

        inline void prepare(PrepareSpecs specs) {
            double sr = specs.sampleRate;
            filters.prepare(specs);
            for (auto& voice : filters) {
                voice.prepare(sr);
                voice.setDrive(7.5f);
            }
        }
        inline void reset() { for (auto& voice : filters) voice.reset(); }

        // Process block: scale inputs using fast tanh, process with filter, output stereo.
        template <typename ProcessDataType>
        inline void process(ProcessDataType& data) {
            auto& fixData = data.template as<ProcessData<getFixChannelAmount()>>();
            auto audioBlock = fixData.toAudioBlock();
            float* leftChannel = audioBlock.getChannelPointer(0);
            float* rightChannel = audioBlock.getChannelPointer(1);
            const int numSamples = static_cast<int>(data.getNumSamples());
            const float tanhConst = TanhHelper::tanh(1.5f);
            for (int i = 0; i < numSamples; ++i) {
                float inL = TanhHelper::tanh(1.5f * leftChannel[i]) / tanhConst;
                float inR = TanhHelper::tanh(1.5f * rightChannel[i]) / tanhConst;
                float outL = 0.f, outR = 0.f;
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
        inline void processFrame(FrameDataType& data) {}

        // Parameter callback: update voices on change.
        template <int P>
        inline void setParameter(double v) {
            if (P == 0) {
                float newVal = static_cast<float>(v);
                if (cutoffFrequency != newVal) {
                    cutoffFrequency = newVal;
                    for (auto& voice : filters)
                        voice.setCutoff(cutoffFrequency);
                }
            }
            else if (P == 1) {
                float newVal = static_cast<float>(v);
                if (resonance != newVal) {
                    resonance = newVal;
                    for (auto& voice : filters)
                        voice.setResonance(resonance);
                }
            }
        }
        inline void createParameters(ParameterDataList& data) {
            parameter::data p1("Cutoff", { 20.0, 4000.0, 0.00001 });
            registerCallback<0>(p1);
            p1.setDefaultValue(1000.0);
            data.add(std::move(p1));
            parameter::data p2("Resonance", { 0.1, 4.3, 0.00001 });
            registerCallback<1>(p2);
            p2.setDefaultValue(1.0);
            data.add(std::move(p2));
        }
        inline void setExternalData(const ExternalData& ed, int index) {}
        inline void handleHiseEvent(HiseEvent& e) {}
    };
}
