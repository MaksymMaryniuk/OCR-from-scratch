using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Model
{
    public class AdaptiveOptimizer : Optimizer
    {
        public double Epsilon { get; set; }


        public AdaptiveOptimizer(double learningRate = 0.01, double decay = 0.001, double epsilon = 1e-8) : base(learningRate, decay)
        {
            Epsilon = epsilon;
        }

        protected void EnsureCacheInitialized(Layer_Dense layer)
        {
            if (layer.WeightCache == null)
            {
                layer.WeightCache = new double[layer.Weights.GetLength(0), layer.Weights.GetLength(1)];
                layer.BiasCache = new double[layer.Biases.Length];

                if (this is Optimizer_Adam && layer.WeightMomentums == null)
                {
                    layer.WeightMomentums = new double[layer.Weights.GetLength(0), layer.Weights.GetLength(1)];
                    layer.BiasMomentums = new double[layer.Biases.Length];
                }
            }
        }
    }
}