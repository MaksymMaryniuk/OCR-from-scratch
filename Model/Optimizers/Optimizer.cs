using Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Model.Optimizers
{
    public abstract class Optimizer
    {
        public float currentLearningRate;
        protected int iteration = 0;

        public float LearningRate { get; set; }

        public float DecayRate { get; set; }

        protected Optimizer(float lr = 0.01F, float decay = 0.001F)
        {
            LearningRate = lr;
            DecayRate = decay;
        }
        public void PreUpdate()
        {
            currentLearningRate = LearningRate * (1.0F / (1.0F + DecayRate * iteration));
        }

        public virtual void Update(Layer_Dense layer)
        {
            ;
        }

        public void PostUpdate()
        {
            iteration++;
        }
    }
}