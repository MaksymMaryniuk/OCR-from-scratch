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
            double sum = 0;

            foreach (double loss in sampleLosses)
            {
                sum += loss;
            }
            return sum / sampleLosses.Length;
        }

        public abstract double[] Forward(double[,] output, double[,] y);
    }
}
