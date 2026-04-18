using Model.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Model.Optimizers
{
    public class Optimizer_Adam : AdaptiveOptimizer
    {
        public double Beta1 { get; set; }
        public double Beta2 { get; set; }

        
        public Optimizer_Adam(double beta1 = 0.9, double beta2 = 0.999, double learningRate = 0.001, double decay = 0.0, double epsilon = 1e-8) :
            base(learningRate, decay, epsilon)
        {
            Beta1 = beta1;
            Beta2 = beta2;
        }

        public override void Update(Layer_Dense layer)
        {
            double beta1Correction = 1 - Math.Pow(Beta1, iteration + 1);
            double beta2Correction = 1 - Math.Pow(Beta2, iteration + 1);
            for (int i = 0; i < layer.Weights.GetLength(0); i++)
            {
                for (int j = 0; j < layer.Weights.GetLength(1); j++)
                {
                    layer.WeightMomentums[i, j] = Beta1 * layer.WeightMomentums[i, j] + (1 - Beta1) * layer.dWeights[i, j];
                    double WMomentumCorrected = layer.WeightMomentums[i, j] / beta1Correction;
                    layer.WeightCache[i, j] = Beta2 * layer.WeightCache[i, j] + (1 - Beta2) * Math.Pow(layer.dWeights[i, j], 2);
                    double WCacheCorrected = layer.WeightCache[i, j] / beta2Correction;
                    layer.Weights[i, j] -= (currentLearningRate * WMomentumCorrected) / (Math.Sqrt(WCacheCorrected) + Epsilon);
                }
            }
            for (int j = 0; j < layer.Biases.Length; j++)
            {
                layer.BiasMomentums[j] = Beta1 * layer.BiasMomentums[j] + (1 - Beta1) * layer.dBiases[j];
                double BMomentumCorrected = layer.BiasMomentums[j] / beta1Correction;
                layer.BiasCache[j] = Beta2 * layer.BiasCache[j] + (1 - Beta2) * Math.Pow(layer.dBiases[j], 2);
                double BCacheCorrected = layer.BiasCache[j] / beta2Correction;
                layer.Biases[j] -= (currentLearningRate * BMomentumCorrected) / (Math.Sqrt(BCacheCorrected) + Epsilon);
            }
        }
    }
}
