using Vionet.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Vionet.Optimizers
{
    public class Optimizer_SGD : Optimizer
    {
        protected float momentum = 0.0F;

        public Optimizer_SGD(float lr = 0.01F, float decay = 0.001F, float momentum = 0.9F) : base(lr, decay)
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
                        float update = (momentum * layer.WeightMomentums[i, j]) - (CurrentLearningRate * layer.dWeights[i, j]);
                        layer.WeightMomentums[i, j] = update;
                        layer.Weights[i, j] += update;
                    }
                }

                for (int j = 0; j < layer.Biases.Length; j++)
                {
                    float update = (momentum * layer.BiasMomentums[j]) - (CurrentLearningRate * layer.dBiases[j]);
                    layer.BiasMomentums[j] = update;
                    layer.Biases[j] += update;
                }
            }
            else
            {
                for (int i = 0; i < layer.Weights.GetLength(0); i++)
                {
                    for (int j = 0; j < layer.Weights.GetLength(1); j++)
                    {
                        layer.Weights[i, j] -= CurrentLearningRate * layer.dWeights[i, j];
                    }
                }

                for (int j = 0; j < layer.Biases.Length; j++)
                {
                    layer.Biases[j] -= CurrentLearningRate * layer.dBiases[j];
                }
            }
        }
    }
}
