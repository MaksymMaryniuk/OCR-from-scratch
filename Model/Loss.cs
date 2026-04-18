using Model.Layers;
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
            return sampleLosses.Average();
        }
        public double Calculate(double[,] output, int[] y)
        {
            double[] sampleLosses = Forward(output, y);
            return sampleLosses.Average();
        }
        public abstract double[] Forward(double[,] output, int[] y);
        public abstract double[] Forward(double[,] output, double[,] y);
        public abstract double[,] Backward(double[,] dvalues, int[] y);

        public double Regularization_Loss(Layer layer)
        {
            double regularization_loss = 0.0;
            if (layer is Layer_Dense denseLayer)
            {
                if (denseLayer.L1W > 0)
                {
                    regularization_loss += denseLayer.L1W * AdditionalMath.SumOfAbsoluteValues(denseLayer.Weights);
                }
                if (denseLayer.L2W > 0)
                {
                    regularization_loss += denseLayer.L2W * AdditionalMath.SumOfSquares(denseLayer.Weights);
                }
                if (denseLayer.L1B > 0)
                {
                    regularization_loss += denseLayer.L1B * AdditionalMath.SumOfAbsoluteValues(denseLayer.Biases);
                }
                if (denseLayer.L2B > 0)
                {
                    regularization_loss += denseLayer.L2B * AdditionalMath.SumOfSquares(denseLayer.Biases);
                }
            }
            return regularization_loss;
        }
    }
}
