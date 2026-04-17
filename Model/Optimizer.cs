using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Model
{
    public abstract class Optimizer
    {
        public double currentLearningRate;
        protected int iteration = 0;

        public double LearningRate { get; set; }

        public double DecayRate { get; set; }

        protected Optimizer(double lr = 0.01, double decay = 0.001)
        {
            LearningRate = lr;
            DecayRate = decay;
        }
        public void PreUpdate()
        {
            currentLearningRate = LearningRate * (1.0 / (1.0 + DecayRate * iteration));
        }

        public virtual void Update(Dense layer)
        {
            ;
        }

        public void PostUpdate()
        {
            iteration++;
        }
    }
}