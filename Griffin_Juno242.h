#pragma once
#include <JuceHeader.h>
#include <cmath>
#include <array>
#include <algorithm>

#ifndef USE_APPROX_TANH
#define USE_APPROX_TANH true
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Compile-time tolerance multiplier: set to 1.0f for the original 3% variation,
// or to a higher value (e.g. 1.5f) to intensify the differences.
#ifndef TOLERANCE_MULTIPLIER
#define TOLERANCE_MULTIPLIER 1.0f
#endif

namespace project {

    using namespace juce;
    using namespace hise;
    using namespace scriptnode;

    // Assume FunctionsClasses::TanhHelper is defined elsewhere.

    //---------------------------------------------------------------------
    // Dual-channel Juno filter voice class with fixed per-stage tolerance.
    // Each voice deterministically selects a candidate configuration (from fixed tables)
    // for its left and right channels so that each voice always has a unique, consistent
    // variation. The candidate values here vary between 1.00 and 1.03 (3% variation).
    // The per-stage tolerances are scaled at compile time using TOLERANCE_MULTIPLIER.
    //---------------------------------------------------------------------
    template <bool UseApproxTanh = USE_APPROX_TANH>
    class JunoFilterStereoDual {
    public:
        JunoFilterStereoDual()
            : cutoff(1000.f), resonance(1.f), sr(44100.0),
            errorThresh(0.000001f)
        {
            // Define candidate tables for the left and right channel multipliers.
            // These values are fixed and chosen to give about 3% variation.
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

            // Use a static counter so that each new voice gets a different configuration.
            static int voiceCounter = 0;
            int myIndex = voiceCounter++;
            int leftConfig = myIndex % candidateLeft.size();
            int rightConfig = (myIndex + 1) % candidateRight.size();

            // Apply the compile-time multiplier to the candidate values.
            for (int i = 0; i < 4; ++i) {
                stageToleranceLeft[i] = 1.f + (candidateLeft[leftConfig][i] - 1.f) * TOLERANCE_MULTIPLIER;
                stageToleranceRight[i] = 1.f + (candidateRight[rightConfig][i] - 1.f) * TOLERANCE_MULTIPLIER;
            }

            // Initialize state arrays.
            for (int i = 0; i < 4; ++i) {
                yL[i] = yL_est[i] = FL[i] = 0.f;
                yR[i] = yR_est[i] = FR[i] = 0.f;
            }
            sL1 = sL2 = sL3 = sL4 = 0.f;
            sR1 = sR2 = sR3 = sR4 = 0.f;
        }

        void prepare(double sampleRate) { sr = sampleRate; }

        void reset() {
            for (int i = 0; i < 4; ++i) {
                yL[i] = 0.f;
                yR[i] = 0.f;
            }
            sL1 = sL2 = sL3 = sL4 = 0.f;
            sR1 = sR2 = sR3 = sR4 = 0.f;
        }

        // Process a stereo sample.
        // inL and inR are the left/right inputs.
        // Returns a pair {left output, right output}.
        std::pair<float, float> processSample(float inL, float inR) {
            float baseG = std::tan(cutoff * M_PI / sr);
            // Compute per-stage g factors using the fixed tolerances.
            float gL[4], gR[4];
            for (int i = 0; i < 4; ++i) {
                gL[i] = baseG * stageToleranceLeft[i];
                gR[i] = baseG * stageToleranceRight[i];
            }

            //--- Process left channel ---
            int iter = 0;
            float residueL = 1e6f;
            while (std::abs(residueL) > errorThresh && iter < 50) {
                for (int i = 0; i < 4; ++i)
                    yL_est[i] = yL[i];

                // Multiply the tanh argument by the stage tolerance.
                float tanh_y1 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh((inL - yL[0] - resonance * yL[3]) * stageToleranceLeft[0]);
                float tanh_y2 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh((yL[0] - yL[1]) * stageToleranceLeft[1]);
                float tanh_y3 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh((yL[1] - yL[2]) * stageToleranceLeft[2]);
                float tanh_y4 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh((yL[2] - yL[3]) * stageToleranceLeft[3]);

                FL[0] = gL[0] * tanh_y1 + sL1 - yL[0];
                FL[1] = gL[1] * tanh_y2 + sL2 - yL[1];
                FL[2] = gL[2] * tanh_y3 + sL3 - yL[2];
                FL[3] = gL[3] * tanh_y4 + sL4 - yL[3];

                float help_y1 = 1.f - tanh_y1 * tanh_y1;
                float help_y2 = 1.f - tanh_y2 * tanh_y2;
                float help_y3 = 1.f - tanh_y3 * tanh_y3;
                float help_y4 = 1.f - tanh_y4 * tanh_y4;

                float jL00 = -gL[0] * help_y1 - 1.f;
                float jL03 = -gL[0] * resonance * help_y1;
                float jL10 = gL[1] * help_y2;
                float jL11 = -gL[1] * help_y2 - 1.f;
                float jL21 = gL[2] * help_y3;
                float jL22 = -gL[2] * help_y3 - 1.f;
                float jL32 = gL[3] * help_y4;
                float jL33 = -gL[3] * help_y4 - 1.f;

                float denL = jL00 * jL11 * jL22 * jL33 - jL03 * jL10 * jL21 * jL32;

                float deltaL0 = (FL[1] * jL03 * jL21 * jL32 - FL[0] * jL11 * jL22 * jL33 - FL[2] * jL03 * jL11 * jL32 + FL[3] * jL03 * jL11 * jL22) / denL;
                float deltaL1 = (FL[0] * jL10 * jL22 * jL33 - FL[1] * jL00 * jL22 * jL33 + FL[2] * jL03 * jL10 * jL32 - FL[3] * jL03 * jL10 * jL22) / denL;
                float deltaL2 = (FL[1] * jL00 * jL21 * jL33 - FL[0] * jL10 * jL21 * jL33 - FL[2] * jL00 * jL11 * jL33 + FL[3] * jL03 * jL10 * jL21) / denL;
                float deltaL3 = (FL[0] * jL10 * jL21 * jL32 - FL[1] * jL00 * jL21 * jL32 + FL[2] * jL00 * jL11 * jL32 - FL[3] * jL00 * jL11 * jL22) / denL;

                yL[0] += deltaL0;
                yL[1] += deltaL1;
                yL[2] += deltaL2;
                yL[3] += deltaL3;

                residueL = yL[3] - yL_est[3];
                iter++;
            }
            sL1 = 2.f * yL[0] - sL1;
            sL2 = 2.f * yL[1] - sL2;
            sL3 = 2.f * yL[2] - sL3;
            sL4 = 2.f * yL[3] - sL4;
            float outL = yL[3];

            //--- Process right channel ---
            int iterR = 0;
            float residueR = 1e6f;
            while (std::abs(residueR) > errorThresh && iterR < 50) {
                for (int i = 0; i < 4; ++i)
                    yR_est[i] = yR[i];

                float tanh_y1 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh((inR - yR[0] - resonance * yR[3]) * stageToleranceRight[0]);
                float tanh_y2 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh((yR[0] - yR[1]) * stageToleranceRight[1]);
                float tanh_y3 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh((yR[1] - yR[2]) * stageToleranceRight[2]);
                float tanh_y4 = FunctionsClasses::TanhHelper<UseApproxTanh>::tanh((yR[2] - yR[3]) * stageToleranceRight[3]);

                FR[0] = gR[0] * tanh_y1 + sR1 - yR[0];
                FR[1] = gR[1] * tanh_y2 + sR2 - yR[1];
                FR[2] = gR[2] * tanh_y3 + sR3 - yR[2];
                FR[3] = gR[3] * tanh_y4 + sR4 - yR[3];

                float help_y1_r = 1.f - tanh_y1 * tanh_y1;
                float help_y2_r = 1.f - tanh_y2 * tanh_y2;
                float help_y3_r = 1.f - tanh_y3 * tanh_y3;
                float help_y4_r = 1.f - tanh_y4 * tanh_y4;

                float jR00 = -gR[0] * help_y1_r - 1.f;
                float jR03 = -gR[0] * resonance * help_y1_r;
                float jR10 = gR[1] * help_y2_r;
                float jR11 = -gR[1] * help_y2_r - 1.f;
                float jR21 = gR[2] * help_y3_r;
                float jR22 = -gR[2] * help_y3_r - 1.f;
                float jR32 = gR[3] * help_y4_r;
                float jR33 = -gR[3] * help_y4_r - 1.f;

                float denR = jR00 * jR11 * jR22 * jR33 - jR03 * jR10 * jR21 * jR32;

                float deltaR0 = (FR[1] * jR03 * jR21 * jR32 - FR[0] * jR11 * jR22 * jR33 - FR[2] * jR03 * jR11 * jR32 + FR[3] * jR03 * jR11 * jR22) / denR;
                float deltaR1 = (FR[0] * jR10 * jR22 * jR33 - FR[1] * jR00 * jR22 * jR33 + FR[2] * jR03 * jR10 * jR32 - FR[3] * jR03 * jR10 * jR22) / denR;
                float deltaR2 = (FR[1] * jR00 * jR21 * jR33 - FR[0] * jR10 * jR21 * jR33 - FR[2] * jR00 * jR11 * jR33 + FR[3] * jR03 * jR10 * jR21) / denR;
                float deltaR3 = (FR[0] * jR10 * jR21 * jR32 - FR[1] * jR00 * jR21 * jR32 + FR[2] * jR00 * jR11 * jR32 - FR[3] * jR00 * jR11 * jR22) / denR;

                yR[0] += deltaR0;
                yR[1] += deltaR1;
                yR[2] += deltaR2;
                yR[3] += deltaR3;

                residueR = yR[3] - yR_est[3];
                iterR++;
            }
            sR1 = 2.f * yR[0] - sR1;
            sR2 = 2.f * yR[1] - sR2;
            sR3 = 2.f * yR[2] - sR3;
            sR4 = 2.f * yR[3] - sR4;
            float outR = yR[3];

            return { outL, outR };
        }

        void setCutoff(float newCutoff) { cutoff = newCutoff; }
        void setResonance(float newResonance) { resonance = newResonance; }

    private:
        double sr;
        float cutoff;
        float resonance;
        float errorThresh;

        // Left channel state.
        float yL[4], yL_est[4], FL[4];
        float sL1, sL2, sL3, sL4;
        float stageToleranceLeft[4];

        // Right channel state.
        float yR[4], yR_est[4], FR[4];
        float sR1, sR2, sR3, sR4;
        float stageToleranceRight[4];

        juce::Random randGen;
    };

    //---------------------------------------------------------------------
    // Polyphonic node for the stereo Juno filter.
    // This node instantiates voices of JunoFilterStereoDual and processes stereo input.
    // It also implements a drive knob that multiplies the input gain so that drive = 1.0 doubles the input.
    //---------------------------------------------------------------------
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
        static constexpr int getFixChannelAmount() { return 2; } // Stereo output.

        static constexpr int NumTables = 0;
        static constexpr int NumSliderPacks = 0;
        static constexpr int NumAudioFiles = 0;
        static constexpr int NumFilters = 0;
        static constexpr int NumDisplayBuffers = 0;

        float cutoffFrequency = 1000.f;
        float resonance = 1.f;
        float drive = 0.f; // Range: 0.0 (no drive) to 1.0 (drive doubles the input)

        SmoothedValue<float> cutoffSmooth;
        SmoothedValue<float> resonanceSmooth;
        SmoothedValue<float> driveSmooth;

        PolyData<JunoFilterStereoDual<USE_APPROX_TANH>, NV> filters;

        void prepare(PrepareSpecs specs) {
            double sampleRate = specs.sampleRate;
            cutoffSmooth.reset(sampleRate, 0.01);
            resonanceSmooth.reset(sampleRate, 0.01);
            driveSmooth.reset(sampleRate, 0.01);
            cutoffSmooth.setCurrentAndTargetValue(cutoffFrequency);
            resonanceSmooth.setCurrentAndTargetValue(resonance);
            driveSmooth.setCurrentAndTargetValue(drive);

            filters.prepare(specs);
            for (auto& voice : filters)
                voice.prepare(sampleRate);
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
                float cVal = cutoffSmooth.getNextValue();
                float rVal = resonanceSmooth.getNextValue();
                float driveVal = driveSmooth.getNextValue();
                // Multiply input gain by (1 + driveVal), so drive = 1.0 doubles the input.
                float inputGain = 1.f + driveVal;

                float inL = leftChannel[i] * inputGain;
                float inR = rightChannel[i] * inputGain;
                float outL = 0.f;
                float outR = 0.f;

                for (auto& voice : filters) {
                    voice.setCutoff(cVal);
                    voice.setResonance(rVal);
                    auto outs = voice.processSample(inL, inR);
                    outL += outs.first;
                    outR += outs.second;
                }
                leftChannel[i] = outL / NV;
                rightChannel[i] = outR / NV;
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
                drive = static_cast<float>(v);
                driveSmooth.setTargetValue(drive);
            }
        }

        void createParameters(ParameterDataList& data) {
            {
                parameter::data p("Cutoff", { 20.0, 20000.0, 0.01 });
                registerCallback<0>(p);
                p.setDefaultValue(1000.0);
                data.add(std::move(p));
            }
            {
                parameter::data p("Resonance", { 0.1, 4.0, 0.01 });
                registerCallback<1>(p);
                p.setDefaultValue(1.0);
                data.add(std::move(p));
            }
            {
                parameter::data p("Drive", { 0.0, 1.0, 0.01 });
                registerCallback<2>(p);
                p.setDefaultValue(0.0);
                data.add(std::move(p));
            }
        }

        void setExternalData(const ExternalData& ed, int index) {}

        void handleHiseEvent(HiseEvent& e) {
            // Handle note events if needed.
        }
    };

} // namespace project
