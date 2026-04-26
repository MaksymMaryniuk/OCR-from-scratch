using Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Model.Optimizers
{
    public class AdaptiveOptimizer : Optimizer
    {
        public float Epsilon { get; set; }


        public AdaptiveOptimizer(float learningRate = 0.01f, float decay = 0.001f, float epsilon = 1e-8f) : base(learningRate, decay)
        {
            Epsilon = epsilon;
        }

        protected void EnsureCacheInitialized(Layer_Dense layer)
        {
            if (layer.WeightCache == null)
            {
                layer.WeightCache = new float[layer.Weights.GetLength(0), layer.Weights.GetLength(1)];
                layer.BiasCache = new float[layer.Biases.Length];

                if (this is Optimizer_Adam && layer.WeightMomentums == null)
                {
                    layer.WeightMomentums = new float[layer.Weights.GetLength(0), layer.Weights.GetLength(1)];
                    layer.BiasMomentums = new float[layer.Biases.Length];
                }
            }
        }
    }
}