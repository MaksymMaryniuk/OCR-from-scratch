using System;
using System.Collections.Generic;
using System.Text;

namespace Model.Layers
{
    public class ActivationSoftmax : Layer
    {
        public override void Forward(float[,] inputs)
        {
            int rows = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            Output = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                float maxInput = float.NegativeInfinity;
                for (int j = 0; j < cols; j++)
                {
                    if (inputs[i, j] > maxInput)
                        maxInput = inputs[i, j];
                }
                float sumExp = 0;
                for (int j = 0; j < cols; j++)
                {
                    sumExp += MathF.Exp(inputs[i, j] - maxInput);
                }
                for (int j = 0; j < cols; j++)
                {
                    Output[i, j] = MathF.Exp(inputs[i, j] - maxInput) / sumExp;
                }
            }
        }
        public override void Backward(float[,] y_true)
        {
            Dinputs = new float[Output.GetLength(0), Output.GetLength(1)];

            for (int i = 0; i < Output.GetLength(0); i++)
            {
                for (int j = 0; j < Output.GetLength(1); j++)
                {
                    Dinputs[i, j] = Output[i, j] - y_true[i, j];
                }
            }
            // operations: Dinputs = A - Y
            // where A - output from softmax, Y - vector of true labels (one-hot encoding)
        }
        public void Backward(int[] y_true)
        {
            Dinputs = new float[Output.GetLength(0), Output.GetLength(1)];
            for (int i = 0; i < Output.GetLength(0); i++)
            {
                for (int j = 0; j < Output.GetLength(1); j++)
                {
                    Dinputs[i, j] = Output[i, j] - (y_true[i] == j ? 1 : 0);
                }
            }
            // operations: Dinputs = A - Y
            // where A - output from softmax, Y - vector of true labels (one-hot encoding)
        }
    }
}
