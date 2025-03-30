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

    // Mono filter using Newton-Raphson iteration.
    // The filter coefficient is cached (cachedBaseG) and updated only when cutoff or sr change.
    class JunoFilterMono {
    public:
        JunoFilterMono()
            : cutoff(1000.f), resonance(1.f), drive(7.5f), sr(44100.0),
            errorThresh(0.000001f), cachedBaseG(std::tan(1000.f * M_PI / 44100.0))
        {
            for (int i = 0; i < 4; ++i)
                y[i] = 0.f;
            s[0] = s[1] = s[2] = s[3] = 0.f;
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
            for (int i = 0; i < 4; ++i)
                y[i] = 0.f;
            s[0] = s[1] = s[2] = s[3] = 0.f;
        }

        // Process one sample using cached filter coefficient.
        inline float processSample(float in) {
            // Use precomputed base coefficient (cachedBaseG) for all stages.
            const float g = cachedBaseG;
            // Generate noise only if input is nonzero.
            const float noise = (NOISE_FEEDBACK_VOLUME > 0.f && std::abs(in) > INPUT_SIGNAL_THRESHOLD) ?
                NOISE_FEEDBACK_VOLUME * (randGen.nextFloat() * 2.f - 1.f) : 0.f;
            // Newton-Raphson iteration (max 20 iterations)
            for (int iter = 0; iter < 20; ++iter) {
                const float prev_y3 = y[3];
                const float nl0 = transistorHelpers::advancedTransistorNonlinearity(in - y[0] - resonance * y[3] + noise, drive);
                const float nl1 = transistorHelpers::advancedTransistorNonlinearity(y[0] - y[1], drive);
                const float nl2 = transistorHelpers::advancedTransistorNonlinearity(y[1] - y[2], drive);
                const float nl3 = transistorHelpers::advancedTransistorNonlinearity(y[2] - y[3], drive);
                const float f0 = g * nl0 + s[0] - y[0];
                const float f1 = g * nl1 + s[1] - y[1];
                const float f2 = g * nl2 + s[2] - y[2];
                const float f3 = g * nl3 + s[3] - y[3];
                const float h0 = 1.f - nl0 * nl0;
                const float h1 = 1.f - nl1 * nl1;
                const float h2 = 1.f - nl2 * nl2;
                const float h3 = 1.f - nl3 * nl3;
                const float j00 = -g * h0 - 1.f;
                const float j03 = -g * resonance * h0;
                const float j10 = g * h1;
                const float j11 = -g * h1 - 1.f;
                const float j21 = g * h2;
                const float j22 = -g * h2 - 1.f;
                const float j32 = g * h3;
                const float j33 = -g * h3 - 1.f;
                const float den = j00 * j11 * j22 * j33 - j03 * j10 * j21 * j32;
                y[0] += (f1 * j03 * j21 * j32 - f0 * j11 * j22 * j33 - f2 * j03 * j11 * j32 + f3 * j03 * j11 * j22) / den;
                y[1] += (f0 * j10 * j22 * j33 - f1 * j00 * j22 * j33 + f2 * j03 * j10 * j32 - f3 * j03 * j10 * j22) / den;
                y[2] += (f1 * j00 * j21 * j33 - f0 * j10 * j21 * j33 - f2 * j00 * j11 * j33 + f3 * j03 * j10 * j21) / den;
                y[3] += (f0 * j10 * j21 * j32 - f1 * j00 * j21 * j32 + f2 * j00 * j11 * j32 - f3 * j00 * j11 * j22) / den;
                if (std::abs(y[3] - prev_y3) <= errorThresh)
                    break;
            }
            s[0] = 2.f * y[0] - s[0];
            s[1] = 2.f * y[1] - s[1];
            s[2] = 2.f * y[2] - s[2];
            s[3] = 2.f * y[3] - s[3];
            return y[3];
        }
    private:
        double sr;
        float cutoff, resonance, drive, errorThresh;
        float cachedBaseG;
        float y[4], s[4];
        juce::Random randGen;
    };

    // Polyphonic mono node.
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
        PolyData<JunoFilterMono, NV> filters;

        // Prepare voices; update sample rate only.
        inline void prepare(PrepareSpecs specs) {
            double sr = specs.sampleRate;
            filters.prepare(specs);
            for (auto& voice : filters) {
                voice.prepare(sr);
                voice.setDrive(7.5f);
            }
        }
        inline void reset() { for (auto& voice : filters) voice.reset(); }

        // Process audio block: apply fast tanh scaling, sum voices, process with filter, copy mono output to both channels.
        template <typename ProcessDataType>
        inline void process(ProcessDataType& data) {
            auto& fixData = data.template as<ProcessData<getFixChannelAmount()>>();
            auto audioBlock = fixData.toAudioBlock();
            float* leftChannel = audioBlock.getChannelPointer(0);
            float* rightChannel = audioBlock.getChannelPointer(1);
            const int numSamples = static_cast<int>(data.getNumSamples());
            const float tanhConst = TanhHelper::tanh(1.5f);
            for (int i = 0; i < numSamples; ++i) {
                float in = TanhHelper::tanh(1.5f * leftChannel[i]) / tanhConst;
                float out = 0.f;
                for (auto& voice : filters)
                    out += voice.processSample(in);
                out /= NV;
                leftChannel[i] = out;
                rightChannel[i] = out;
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
            parameter::data p2("Resonance", { 0.1, 4.0, 0.00001 });
            registerCallback<1>(p2);
            p2.setDefaultValue(0.8);
            data.add(std::move(p2));
        }
        inline void setExternalData(const ExternalData& ed, int index) {}
        inline void handleHiseEvent(HiseEvent& e) {}
    };
}
