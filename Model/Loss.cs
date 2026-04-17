using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public abstract class Loss
    {
        public double Calculate(double[,] output, double[,] y)
        {
            double[] sampleLosses = Forward(output, y);
            return Mean(sampleLosses);
        }
        public double Calculate(double[,] output, int[] y)
        {
            double[] sampleLosses = Forward(output, y);
            return Mean(sampleLosses);
        }
        protected double Mean(double[] values)
        {
            double sum = 0;

            foreach (double value in values)
            {
                sum += value;
            }

            return sum / values.Length;
        }
        public abstract double[] Forward(double[,] output, int[] y);
        public abstract double[] Forward(double[,] output, double[,] y);
    }
}
