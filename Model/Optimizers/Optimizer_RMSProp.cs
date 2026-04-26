using Model.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Model.Optimizers
{
    public class Optimizer_RMSProp : AdaptiveOptimizer
    {

        public float Rho { get; set; }
        public Optimizer_RMSProp(float rho, float learningRate = 1.0F, float decay = 0.0F, float epsilon = 1e-8F) :
            base(learningRate, decay, epsilon)
        {
            Rho = rho;
        }

        public override void Update(Layer_Dense layer)
        {
            for (int i = 0; i < layer.Weights.GetLength(0); i++)
            {
                for (int j = 0; j < layer.Weights.GetLength(1); j++)
                {
                    layer.WeightMomentums[i, j] = Rho * layer.WeightMomentums[i, j] + (1 - Rho) * MathF.Pow(layer.dWeights[i, j], 2);
                    layer.Weights[i, j] -= (currentLearningRate * layer.dWeights[i, j]) / (MathF.Sqrt(layer.WeightMomentums[i, j]) + Epsilon);
                }
            }
            for (int j = 0; j < layer.Biases.Length; j++)
            {
                layer.BiasMomentums[j] = Rho * layer.BiasMomentums[j] + (1 - Rho) * MathF.Pow(layer.dBiases[j], 2);
                layer.Biases[j] -= (currentLearningRate * layer.dBiases[j]) / (MathF.Sqrt(layer.BiasMomentums[j]) + Epsilon);
            }
        }
    }
}
