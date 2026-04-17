using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public class ActivationReLU : Layer
    {
        public override void Forward(double[,] inputs) 
        {
            Inputs = inputs;
            Output = new double[inputs.GetLength(0), inputs.GetLength(1)];
            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                for (int j = 0; j < inputs.GetLength(1); j++)
                {
                    Output[i, j] = Math.Max(0, inputs[i, j]);
                }
            }
        }
        public override double[,] Backward(double[,] dA)
        {
            // operations: dZ = dA * (Z > 0)
            // where dA - grad due to previous layer (or from derivitave loss to respect of ), Z - input to ReLU

            double[,] dZ = new double[dA.GetLength(0), dA.GetLength(1)];

            for (int i = 0; i < dA.GetLength(0); i++)
            {
                for (int j = 0; j < dA.GetLength(1); j++)
                {
                    dZ[i, j] = Inputs[i, j] > 0 ? dA[i, j] : 0.001 * dA[i, j];
                }
            }
            return dZ;
        }
    }
}
