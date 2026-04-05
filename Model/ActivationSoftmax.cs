using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public class ActivationSoftmax
    {
        public double[,] Output { get; set; }
        public void Forward(double[,] inputs)
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

    }
}
