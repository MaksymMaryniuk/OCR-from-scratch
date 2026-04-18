using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public class Layer_Dropout : Layer
    {
        public double Rate { get; set; }
        public double KeepRate { get; set; }
        public double[,] Mask { get; set; }
        private Random rand = new Random();
        public Layer_Dropout(double rate)
        {
            Rate = rate;
            KeepRate = 1- rate;
        }

        public override void Forward(double[,] inputs)
        {
            Inputs = inputs;

            Mask = new double[inputs.GetLength(0), inputs.GetLength(1)];
            Output = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                for (int j = 0; j < inputs.GetLength(1); j++)
                {
                    Mask[i, j] = rand.NextDouble() < KeepRate ? 1.0 : 0.0;
                    Mask[i, j] /= KeepRate;
                    Output[i, j] = Mask[i, j] * inputs[i, j];
                }
            }
        }
        public override void Backward(double[,] dvalues)
        {
            Dinputs = new double[dvalues.GetLength(0), dvalues.GetLength(1)];

            for (int i = 0; i < dvalues.GetLength(0); i++)
            {
                for (int j = 0; j < dvalues.GetLength(1); j++)
                {
                    Dinputs[i, j] = Mask[i, j] * dvalues[i, j];
                }
            }
        }
    }
}
