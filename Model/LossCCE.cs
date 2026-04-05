using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public class LossCCE : Loss
    {
        public double[] Outputs { get; set; }
        public override double[] Forward(double[,] y_pred, double[,] y_true)
        {
            double epsilon = 1e-7;

            for (int i = 0; i < y_pred.GetLength(0); i++)
            {
                for (int j = 0; j < y_pred.GetLength(1); j++)
                {
                    y_pred[i, j] = Math.Clamp(y_pred[i, j], epsilon, 1 - epsilon);
                }
            }

            double[] sampleLosses = new double[y_pred.GetLength(0)];

            // Calculate CCE for scalar values
            for (int i = 0; i < y_pred.GetLength(0); i++)
            {
                for (int j = 0; j < y_true.GetLength(1); j++)
                {
                    if (y_true[i, j] == 1)
                    {
                        double correct_confidence = y_pred[i, j];
                        sampleLosses[i] = -Math.Log(correct_confidence);
                        break; // Assuming one-hot encoding, we can break after finding the correct class
                    }
                }

            }
            return sampleLosses;
        }
        public double[,] Backward(double[,] softmax_output, double[,] y_true)
        {
            double[,] dA = new double[softmax_output.GetLength(0), softmax_output.GetLength(1)];

            for (int i = 0; i < softmax_output.GetLength(0); i++)
            {
                for (int j = 0; j < softmax_output.GetLength(1); j++)
                {
                    dA[i, j] = softmax_output[i, j] - y_true[i, j];
                }
            }
            // operations: dA = A - Y
            // where A - output from softmax, Y - vector of true labels (one-hot encoding)
            return dA;
        }

    }
}