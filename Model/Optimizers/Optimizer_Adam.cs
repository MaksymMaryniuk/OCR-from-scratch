using Model.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Model.Optimizers
{
    public class Optimizer_Adam : AdaptiveOptimizer
    {
        public float Beta1 { get; set; }
        public float Beta2 { get; set; }

        
        public Optimizer_Adam(float beta1 = 0.9F, float beta2 = 0.999F, float learningRate = 0.001F, float decay = 0.0F, float epsilon = 1e-8F) :
            base(learningRate, decay, epsilon)
        {
            Beta1 = beta1;
            Beta2 = beta2;
        }

        public override void Update(Layer_Dense layer)
        {
            float beta1Correction = 1 - MathF.Pow(Beta1, iteration + 1);
            float beta2Correction = 1 - MathF.Pow(Beta2, iteration + 1);
            for (int i = 0; i < layer.Weights.GetLength(0); i++)
            {
                for (int j = 0; j < layer.Weights.GetLength(1); j++)
                {
                    layer.WeightMomentums[i, j] = Beta1 * layer.WeightMomentums[i, j] + (1 - Beta1) * layer.dWeights[i, j];
                    float WMomentumCorrected = layer.WeightMomentums[i, j] / beta1Correction;
                    layer.WeightCache[i, j] = Beta2 * layer.WeightCache[i, j] + (1 - Beta2) * MathF.Pow(layer.dWeights[i, j], 2);
                    float WCacheCorrected = layer.WeightCache[i, j] / beta2Correction;
                    layer.Weights[i, j] -= (currentLearningRate * WMomentumCorrected) / (MathF.Sqrt(WCacheCorrected) + Epsilon);
                }
            }
            for (int j = 0; j < layer.Biases.Length; j++)
            {
                layer.BiasMomentums[j] = Beta1 * layer.BiasMomentums[j] + (1 - Beta1) * layer.dBiases[j];
                float BMomentumCorrected = layer.BiasMomentums[j] / beta1Correction;
                layer.BiasCache[j] = Beta2 * layer.BiasCache[j] + (1 - Beta2) * MathF.Pow(layer.dBiases[j], 2);
                float BCacheCorrected = layer.BiasCache[j] / beta2Correction;
                layer.Biases[j] -= (currentLearningRate * BMomentumCorrected) / (MathF.Sqrt(BCacheCorrected) + Epsilon);
            }
        }
    }
}
