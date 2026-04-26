using Model.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public abstract class Loss
    {
        public float Calculate(float[,] output, float[,] y)
        {
            float[] sampleLosses = Forward(output, y);
            return sampleLosses.Average();
        }
        public float Calculate(float[,] output, int[] y)
        {
            float[] sampleLosses = Forward(output, y);
            return sampleLosses.Average();
        }
        public abstract float[] Forward(float[,] output, int[] y);
        public abstract float[] Forward(float[,] output, float[,] y);
        public abstract float[,] Backward(float[,] dvalues, int[] y);

        public float Regularization_Loss(Layer layer)
        {
            float regularization_loss = 0.0F;
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
