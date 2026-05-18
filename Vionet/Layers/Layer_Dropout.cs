using System;
using System.Collections.Generic;
using System.Text;

namespace Vionet.Layers
{
    public class Layer_Dropout : Layer
    {
        public override bool IsTrainable => false;
        public override string Type => "DROPOUT";
        public float Rate { get; private set; }
        private readonly float _keepRate;
        private float[,] _mask;
        private Random rand = new Random();
        public Layer_Dropout(float rate)
        {
            Rate = rate;
            _keepRate = 1 - rate;
        }

        internal override void Forward(float[,] inputs)
        {
            Inputs = inputs;

            if (!IsTraining)
            {
                Output = (float[,])inputs.Clone();
                return;
            }

            _mask = new float[inputs.GetLength(0), inputs.GetLength(1)];
            Output = new float[inputs.GetLength(0), inputs.GetLength(1)];

            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                for (int j = 0; j < inputs.GetLength(1); j++)
                {
                    _mask[i, j] = (float)(rand.NextDouble() < _keepRate ? 1.0 : 0.0);
                    _mask[i, j] /= _keepRate;
                    Output[i, j] = _mask[i, j] * inputs[i, j];
                }
            }
        }
        internal override void Backward(float[,] dvalues)
        {
            Dinputs = new float[dvalues.GetLength(0), dvalues.GetLength(1)];

            for (int i = 0; i < dvalues.GetLength(0); i++)
            {
                for (int j = 0; j < dvalues.GetLength(1); j++)
                {
                    Dinputs[i, j] = _mask[i, j] * dvalues[i, j];
                }
            }
        }
    }
}
