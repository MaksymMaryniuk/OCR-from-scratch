using System;
using System.Collections.Generic;
using System.Text;

namespace Model.Layers
{
    public class Layer_Dropout : Layer
    {
        public float Rate { get; set; }
        public float KeepRate { get; set; }
        public float[,] Mask { get; set; }
        private Random rand = new Random();
        public Layer_Dropout(float rate)
        {
            Rate = rate;
            KeepRate = 1- rate;
        }

        public override void Forward(float[,] inputs)
        {
            Inputs = inputs;

            Mask = new float[inputs.GetLength(0), inputs.GetLength(1)];
            Output = new float[inputs.GetLength(0), inputs.GetLength(1)];

            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                for (int j = 0; j < inputs.GetLength(1); j++)
                {
                    Mask[i, j] = (float)(rand.NextDouble() < KeepRate ? 1.0 : 0.0);
                    Mask[i, j] /= KeepRate;
                    Output[i, j] = Mask[i, j] * inputs[i, j];
                }
            }
        }
        public override void Backward(float[,] dvalues)
        {
            Dinputs = new float[dvalues.GetLength(0), dvalues.GetLength(1)];

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
