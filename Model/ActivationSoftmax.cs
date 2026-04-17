using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public class ActivationSoftmax : Layer
    {
        public override void Forward(double[,] inputs)
        {
            int rows = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            Output = new double[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                double maxInput = double.NegativeInfinity;
                for (int j = 0; j < cols; j++)
                {
                    if (inputs[i, j] > maxInput)
                        maxInput = inputs[i, j];
                }
                double sumExp = 0;
                for (int j = 0; j < cols; j++)
                {
                    sumExp += Math.Exp(inputs[i, j] - maxInput);
                }
                for (int j = 0; j < cols; j++)
                {
                    Output[i, j] = Math.Exp(inputs[i, j] - maxInput) / sumExp;
                }
            }
        }
        public override double[,] Backward(double[,] y_true)
        {
            double[,] dA = new double[Output.GetLength(0), Output.GetLength(1)];

            for (int i = 0; i < Output.GetLength(0); i++)
            {
                for (int j = 0; j < Output.GetLength(1); j++)
                {
                    dA[i, j] = Output[i, j] - y_true[i, j];
                }
            }
            // operations: dA = A - Y
            // where A - output from softmax, Y - vector of true labels (one-hot encoding)
            return dA;
        }
        public double[,] Backward(int[] y_true)
        {
            double[,] dA = new double[Output.GetLength(0), Output.GetLength(1)];
            for (int i = 0; i < Output.GetLength(0); i++)
            {
                for (int j = 0; j < Output.GetLength(1); j++)
                {
                    dA[i, j] = Output[i, j] - (y_true[i] == j ? 1 : 0);
                }
            }
            // operations: dA = A - Y
            // where A - output from softmax, Y - vector of true labels (one-hot encoding)
            return dA;
        }
    }
}
