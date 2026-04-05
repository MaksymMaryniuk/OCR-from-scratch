using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public class Optimizer_SGD
    {
        public double LearningRate { get; set; }

        public Optimizer_SGD(double lr = 0.01)
        {
            LearningRate = lr;
        }

        public void Update(Layer layer)
        {
            for (int i = 0; i < layer.Weights.GetLength(0); i++)
            {
                for (int j = 0; j < layer.Weights.GetLength(1); j++)
                {
                    layer.Weights[i, j] -= LearningRate * layer.dWeights[i, j];
                }
            }

            for (int j = 0; j < layer.Biases.Length; j++)
            {
                layer.Biases[j] -= LearningRate * layer.dBiases[j];
            }
        }
    }
}
