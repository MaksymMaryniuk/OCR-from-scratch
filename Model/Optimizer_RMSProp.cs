using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public class Optimizer_RMSProp : AdaptiveOptimizer
    {

        public double Rho { get; set; }
        public Optimizer_RMSProp(double rho, double learningRate = 1.0, double decay = 0.0, double epsilon = 1e-8) :
            base(learningRate, decay, epsilon)
        {
            Rho = rho;
        }

        public override void Update(Dense layer)
        {
            for (int i = 0; i < layer.Weights.GetLength(0); i++)
            {
                for (int j = 0; j < layer.Weights.GetLength(1); j++)
                {
                    layer.WeightMomentums[i, j] = Rho * layer.WeightMomentums[i, j] + (1 - Rho) * Math.Pow(layer.dWeights[i, j], 2);
                    layer.Weights[i, j] -= (currentLearningRate * layer.dWeights[i, j]) / (Math.Sqrt(layer.WeightMomentums[i, j]) + Epsilon);
                }
            }
            for (int j = 0; j < layer.Biases.Length; j++)
            {
                layer.BiasMomentums[j] = Rho * layer.BiasMomentums[j] + (1 - Rho) * Math.Pow(layer.dBiases[j], 2);
                layer.Biases[j] -= (currentLearningRate * layer.dBiases[j]) / (Math.Sqrt(layer.BiasMomentums[j]) + Epsilon);
            }
        }
    }
}
