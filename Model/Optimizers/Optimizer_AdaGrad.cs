using Microsoft.VisualBasic;
using Model.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Model.Optimizers
{
    public class Optimizer_AdaGrad : AdaptiveOptimizer
    {
        public Optimizer_AdaGrad(double learningRate = 1.0, double decay = 0.0, double epsilon = 1e-8) : base(learningRate, decay, epsilon)
        {
        }

        public override void Update(Layer_Dense layer)
        {
            for (int i = 0; i < layer.Weights.GetLength(0); i++)
            {
                for (int j = 0; j < layer.Weights.GetLength(1); j++)
                {
                    layer.WeightMomentums[i, j] += Math.Pow(layer.dWeights[i, j], 2);
                    layer.Weights[i, j] -= (currentLearningRate * layer.dWeights[i, j]) / (Math.Sqrt(layer.WeightMomentums[i, j]) + Epsilon);
                }
            }
            for (int j = 0; j < layer.Biases.Length; j++)
            {
                layer.BiasMomentums[j] += Math.Pow(layer.dBiases[j], 2);
                layer.Biases[j] -= (currentLearningRate * layer.dBiases[j]) / (Math.Sqrt(layer.BiasMomentums[j]) + Epsilon);
            }
        }
    }
}


