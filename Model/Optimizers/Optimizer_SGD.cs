using Model.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Model.Optimizers
{
    public class Optimizer_SGD : Optimizer
    {
        protected double momentum = 0.0;

        public Optimizer_SGD(double lr = 0.01, double decay = 0.001, double momentum = 0.9) : base(lr, decay)
        {
            this.momentum = momentum;
        }
        public override void Update(Layer_Dense layer)
        {
            if (momentum != 0)
            {
                for (int i = 0; i < layer.Weights.GetLength(0); i++)
                {
                    for (int j = 0; j < layer.Weights.GetLength(1); j++)
                    {
                        layer.WeightMomentums[i, j] = (momentum * layer.WeightMomentums[i, j]) - (currentLearningRate * layer.dWeights[i, j]);
                        layer.Weights[i, j] += (momentum * layer.WeightMomentums[i, j]) - (currentLearningRate * layer.dWeights[i, j]);
                    }
                }

                for (int j = 0; j < layer.Biases.Length; j++)
                {
                    layer.BiasMomentums[j] = (momentum * layer.BiasMomentums[j]) - (currentLearningRate * layer.dBiases[j]);
                    layer.Biases[j] += (momentum * layer.BiasMomentums[j]) - (currentLearningRate * layer.dBiases[j]);
                }
            }
            else
            {
                currentLearningRate = LearningRate * (1.0 / (1.0 + DecayRate * iteration));
                for (int i = 0; i < layer.Weights.GetLength(0); i++)
                {
                    for (int j = 0; j < layer.Weights.GetLength(1); j++)
                    {
                        layer.Weights[i, j] -= currentLearningRate * layer.dWeights[i, j];
                    }
                }

                for (int j = 0; j < layer.Biases.Length; j++)
                {
                    layer.Biases[j] -= currentLearningRate * layer.dBiases[j];
                }
            }
        }
    }
}
