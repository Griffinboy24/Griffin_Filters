#pragma once
#include <JuceHeader.h>
#include <cmath>
#include "src/eigen-master/Eigen/Dense"

// Griffin Generalized Filter based on Werner & McClellan,
// "Moog Ladder Filter Generalizations Based on State Variable Filters"
// Proceedings of DAFx2020 (pages 70-77)
// (This code is open source under the CC Attribution license.)

namespace project
{
    using namespace juce;
    using namespace hise;
    using namespace scriptnode;

    template <int NV>
    struct Griffin_Moog24 : public data::base
    {
        SNEX_NODE(Griffin_Moog24);

        struct MetadataClass
        {
            SN_NODE_ID("Griffin_Moog24");
        };

        static constexpr bool isModNode() { return false; }
        static constexpr bool isPolyphonic() { return NV > 1; }
        static constexpr bool hasTail() { return false; }
        static constexpr bool isSuspendedOnSilence() { return false; }
        static constexpr int  getFixChannelAmount() { return 2; }

        static constexpr int  NumTables = 0;
        static constexpr int  NumSliderPacks = 0;
        static constexpr int  NumAudioFiles = 0;
        static constexpr int  NumFilters = 0;
        static constexpr int  NumDisplayBuffers = 0;

        // Outer-level parameters and smoothing objects
        float cutoffFrequency = 1000.0f;
        float globalFeedback = 0.0f;
        float damping = 1.0f;

        float keytrackAmount = 1.0f;
        float sampleRate = 44100.0f;

        SmoothedValue<float> cutoffSmooth;
        SmoothedValue<float> feedbackSmooth;
        SmoothedValue<float> keytrackSmooth;
        SmoothedValue<float> dampingSmooth;

        void prepare(PrepareSpecs specs)
        {
            sampleRate = specs.sampleRate;
            filtersLeft.prepare(specs);
            filtersRight.prepare(specs);

            for (auto& fl : filtersLeft)
                fl.prepare(sampleRate);
            for (auto& fr : filtersRight)
                fr.prepare(sampleRate);

            // Initialize per-sample smoothing (10ms ramp time)
            cutoffSmooth.reset(sampleRate, 0.01);
            feedbackSmooth.reset(sampleRate, 0.01);
            keytrackSmooth.reset(sampleRate, 0.01);
            dampingSmooth.reset(sampleRate, 0.01);

            cutoffSmooth.setCurrentAndTargetValue(cutoffFrequency);
            feedbackSmooth.setCurrentAndTargetValue(globalFeedback);
            keytrackSmooth.setCurrentAndTargetValue(keytrackAmount);
            dampingSmooth.setCurrentAndTargetValue(damping);
        }

        void reset()
        {
            for (auto& fl : filtersLeft)
                fl.reset();
            for (auto& fr : filtersRight)
                fr.reset();
        }

        template <typename ProcessDataType>
        void process(ProcessDataType& data)
        {
            auto& fixData = data.template as<ProcessData<getFixChannelAmount()>>();
            auto audioBlock = fixData.toAudioBlock();
            float* leftChannelData = audioBlock.getChannelPointer(0);
            float* rightChannelData = audioBlock.getChannelPointer(1);
            int numSamples = static_cast<int>(data.getNumSamples());

            for (int i = 0; i < numSamples; ++i)
            {
                // Get per-sample smoothed parameters
                float cVal = cutoffSmooth.getNextValue();
                float fbVal = feedbackSmooth.getNextValue();
                float ktVal = keytrackSmooth.getNextValue();
                float dVal = dampingSmooth.getNextValue();

                // Update all voices with current parameters
                for (auto& fl : filtersLeft)
                {
                    fl.setCutoff(cVal);
                    fl.setResonance(fbVal);
                    fl.setKeytrack(ktVal);
                    fl.setDamping(dVal);
                    fl.applyChangesIfNeeded();
                }
                for (auto& fr : filtersRight)
                {
                    fr.setCutoff(cVal);
                    fr.setResonance(fbVal);
                    fr.setKeytrack(ktVal);
                    fr.setDamping(dVal);
                    fr.applyChangesIfNeeded();
                }

                // Process the sample for each voice in series
                float inL = leftChannelData[i];
                float inR = rightChannelData[i];

                for (auto& fl : filtersLeft)
                    inL = fl.processSample(inL);
                for (auto& fr : filtersRight)
                    inR = fr.processSample(inR);

                leftChannelData[i] = inL;
                rightChannelData[i] = inR;
            }
        }

        template <typename FrameDataType>
        void processFrame(FrameDataType& data) {}

        class AudioEffect
        {
        public:
            AudioEffect() = default;

            void prepare(float fs)
            {
                sampleRate = fs;
                baseCutoff = 1000.0f;
                globalFeedback = 0.0f;
                damping = 1.0f;
                storedNote = 60;
                gamma0 = -1.0f; // Set to -1 so that with r = 1 the model recovers the Moog
                reset();
                dirtyFlags = 0;
                updateAll();
            }

            void reset()
            {
                x = Eigen::Vector4f::Zero();
            }

            enum Dirty : uint32_t
            {
                changedCutoff = 1 << 0,
                changedFeedback = 1 << 1,
                changedDamping = 1 << 2,
                changedKeytrack = 1 << 3,
                changedNote = 1 << 4
            };

            inline void setCutoff(float c)
            {
                baseCutoff = c;
                dirtyFlags |= changedCutoff;
            }
            inline void setResonance(float fb)  // now globalFeedback (kHat)
            {
                globalFeedback = fb;
                dirtyFlags |= changedFeedback;
            }
            inline void setDamping(float d)
            {
                damping = d;
                dirtyFlags |= changedDamping;
            }
            inline void setKeytrack(float kt)
            {
                keytrackAmount = kt;
                dirtyFlags |= changedKeytrack;
            }
            inline void setNoteNumber(int n)
            {
                storedNote = n;
                dirtyFlags |= changedNote;
            }
            inline void applyChangesIfNeeded()
            {
                if (dirtyFlags != 0)
                    updateAll();
            }

            // Process a single sample using TPT discretization
            inline float processSample(float input)
            {
                Eigen::Vector4f temp = N * x + gB * input;
                Eigen::Vector4f newX = MInv * temp;
                float out = C.dot(newX) * gainComp;
                x = newX;
                return out;
            }

        private:
            inline void updateAll()
            {
                // Compute effective cutoff with keytracking
                float semitones = (static_cast<float>(storedNote) - 60.0f) * keytrackAmount;
                float noteFactor = std::exp2f(0.0833333f * semitones);  // equivalent to exp2((note-60)/12)
                float fc = baseCutoff * noteFactor;
                if (fc < 20.0f)
                    fc = 20.0f;
                float limit = 0.49f * sampleRate;
                if (fc > limit)
                    fc = limit;

                // TPT warping: g = tan(pi*(fc/sampleRate))
                float norm = fc / sampleRate;
                g = std::tan(MathConstants<float>::pi * norm);

                // Build continuous-time state-space for the generalized filter.
                // Following the paper, let r = damping and kHat = globalFeedback:
                //   A = [ -2*r,    1,     0,   4*kHat*r*r;
                //         -1,      0,     0,   0;
                //          0,     -1,   -2*r,  1;
                //          0,      0,    -1,    0 ]
                float r = damping;
                float kHat = globalFeedback;

                Acont << -2.0f * r, 1.0f, 0.0f, 4.0f * kHat * r * r,
                    -1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, -1.0f, -2.0f * r, 1.0f,
                    0.0f, 0.0f, -1.0f, 0.0f;
                Bcont << 1.0f, 0.0f, 0.0f, 0.0f;
                Ccont << 0.0f, 0.0f, 0.0f, -gamma0;  // gamma0 is -1

                // Discrete-time matrices via TPT:
                Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
                M = I - g * Acont;
                N = I + g * Acont;
                gB = g * Bcont;
                MInv = M.inverse();

                C = Ccont;

                // Gain compensation 
                gainComp = -gamma0 / (1.0f + 4.0f * kHat * r * r);

                dirtyFlags = 0;
            }

            float sampleRate = 44100.0f;
            float baseCutoff = 1000.0f;
            float globalFeedback = 0.0f; // kHat
            float damping = 1.0f;        // r
            float keytrackAmount = 1.0f;
            int storedNote = 60;
            float gamma0 = -1.0f;  // output gain factor

            float g = 0.0f;
            float gainComp = 1.0f;
            uint32_t dirtyFlags = 0;

            Eigen::Matrix4f Acont = Eigen::Matrix4f::Zero();
            Eigen::Vector4f Bcont = Eigen::Vector4f::Zero();
            Eigen::Vector4f Ccont = Eigen::Vector4f::Zero();
            Eigen::Matrix4f M = Eigen::Matrix4f::Zero();
            Eigen::Matrix4f N = Eigen::Matrix4f::Zero();
            Eigen::Matrix4f MInv = Eigen::Matrix4f::Zero();
            Eigen::Vector4f gB = Eigen::Vector4f::Zero();
            Eigen::Vector4f C = Eigen::Vector4f::Zero();
            Eigen::Vector4f x = Eigen::Vector4f::Zero();
        };

        template <int P>
        void setParameter(double val)
        {
            if (P == 0)
            {
                cutoffFrequency = static_cast<float>(val);
                cutoffSmooth.setTargetValue(cutoffFrequency);
                for (auto& fl : filtersLeft)
                {
                    fl.setCutoff(cutoffFrequency);
                    fl.applyChangesIfNeeded();
                }
                for (auto& fr : filtersRight)
                {
                    fr.setCutoff(cutoffFrequency);
                    fr.applyChangesIfNeeded();
                }
            }
            else if (P == 1)
            {
                globalFeedback = static_cast<float>(val);
                feedbackSmooth.setTargetValue(globalFeedback);
                for (auto& fl : filtersLeft)
                {
                    fl.setResonance(globalFeedback);
                    fl.applyChangesIfNeeded();
                }
                for (auto& fr : filtersRight)
                {
                    fr.setResonance(globalFeedback);
                    fr.applyChangesIfNeeded();
                }
            }
            else if (P == 2)
            {
                keytrackAmount = static_cast<float>(val);
                keytrackSmooth.setTargetValue(keytrackAmount);
                for (auto& fl : filtersLeft)
                {
                    fl.setKeytrack(keytrackAmount);
                    fl.applyChangesIfNeeded();
                }
                for (auto& fr : filtersRight)
                {
                    fr.setKeytrack(keytrackAmount);
                    fr.applyChangesIfNeeded();
                }
            }
            else if (P == 3)
            {
                damping = static_cast<float>(val);
                dampingSmooth.setTargetValue(damping);
                for (auto& fl : filtersLeft)
                {
                    fl.setDamping(damping);
                    fl.applyChangesIfNeeded();
                }
                for (auto& fr : filtersRight)
                {
                    fr.setDamping(damping);
                    fr.applyChangesIfNeeded();
                }
            }
        }

        void createParameters(ParameterDataList& data)
        {
            {
                parameter::data p("Cutoff Frequency", { 20.0, 20000.0, 1.0 });
                registerCallback<0>(p);
                p.setDefaultValue(1000.0f);
                data.add(std::move(p));
            }
            {
                parameter::data p("Global Feedback", { 0.0, 1.0, 0.01 });
                registerCallback<1>(p);
                p.setDefaultValue(0.0f);
                data.add(std::move(p));
            }
            {
                parameter::data p("Keytrack Amount", { -1.0, 1.0, 0.01 });
                registerCallback<2>(p);
                p.setDefaultValue(0.0f);
                data.add(std::move(p));
            }
            {
                parameter::data p("Damping", { 0.5, 1.5, 0.01 });
                registerCallback<3>(p);
                p.setDefaultValue(1.1f);
                data.add(std::move(p));
            }
        }

        void setExternalData(const ExternalData& data, int index) {}

        void handleHiseEvent(HiseEvent& e)
        {
            if (e.isNoteOn())
            {
                filtersLeft.get().setNoteNumber(e.getNoteNumber());
                filtersLeft.get().applyChangesIfNeeded();

                filtersRight.get().setNoteNumber(e.getNoteNumber());
                filtersRight.get().applyChangesIfNeeded();
            }
        }

    private:
        PolyData<AudioEffect, NV> filtersLeft;
        PolyData<AudioEffect, NV> filtersRight;
    };
}
