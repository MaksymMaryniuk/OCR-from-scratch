using Microsoft.VisualBasic;
using Vionet.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Vionet.Optimizers
{
    public class Optimizer_AdaGrad : AdaptiveOptimizer
    {
        public Optimizer_AdaGrad(float learningRate = 1.0F, float decay = 0.0F, float epsilon = 1e-8F) : base(learningRate, decay, epsilon)
        {
        }

        public override void Update(Layer_Dense layer)
        {
            for (int i = 0; i < layer.Weights.GetLength(0); i++)
            {
                for (int j = 0; j < layer.Weights.GetLength(1); j++)
                {
                    layer.WeightMomentums[i, j] += layer.dWeights[i, j] * layer.dWeights[i, j];

                    layer.Weights[i, j] -= (CurrentLearningRate * layer.dWeights[i, j]) /
                                           (MathF.Sqrt(layer.WeightMomentums[i, j]) + Epsilon);
                }
            }

            for (int j = 0; j < layer.Biases.Length; j++)
            {
                layer.BiasMomentums[j] += layer.dBiases[j] * layer.dBiases[j];
                layer.Biases[j] -= (CurrentLearningRate * layer.dBiases[j]) /
                                   (MathF.Sqrt(layer.BiasMomentums[j]) + Epsilon);
            }
        }
    }
}


