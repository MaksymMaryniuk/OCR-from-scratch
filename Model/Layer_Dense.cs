using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public class Layer_Dense : Layer
    {
        public double[,] Weights { get; set; }
        public double[] Biases { get; set; }
        public double[,] dWeights { get; set; }
        public double[] dBiases { get; set; }
        public double[,] WeightMomentums { get; set; }
        public double[] BiasMomentums { get; set; }
        public double[,] WeightCache { get; set; }
        public double[] BiasCache { get; set; }
        public double L1W { get; set; }
        public double L2W { get; set; }
        public double L1B { get; set; }
        public double L2B { get; set; }


        public Layer_Dense(int num_inputs, int num_neurons, double l1w = 0.0, double l2w = 0.0, double l1b = 0.0, double l2b = 0.0)
        {
            Weights = new double[num_inputs, num_neurons];
            Biases = new double[num_neurons];

            WeightMomentums = new double[num_inputs, num_neurons];
            BiasMomentums = new double[num_neurons];
            WeightCache = new double[num_inputs, num_neurons];
            BiasCache = new double[num_neurons];
            L1W = l1w;
            L2W = l2w;
            L1B = l1b;
            L2B = l2b;
            AdditionalMath.Matrix_filler(Weights);
            AdditionalMath.Vector_filler(Biases);
        }
        public override void Forward(double[,] X)
        {
            Output = new double[X.GetLength(0), Weights.GetLength(1)];
            Inputs = X;
            Output = AdditionalMath.Matrix_Multiplier(X, Weights);

            for (int i = 0; i < Output.GetLength(0); i++)
            {
                for (int j = 0; j < Output.GetLength(1); j++)
                {
                    Output[i, j] += Biases[j];
                }
            }
        }
        public override void Backward(double[,] dZ)
        {
            /*operations: dW = X^T * dZ
                          db = sum(dZ)
                          dX = dZ * W^T
             where X - inputs, dZ - grad due to output layer, W - weights of layer,
             dW - grad due to weights,
             db - grad due to biases,
             dX - grad due to inputs
            */


            //Gradients on parameters
            dWeights = AdditionalMath.Matrix_Multiplier(AdditionalMath.Transpose(Inputs), dZ);
            dBiases = new double[dZ.GetLength(1)];

            for (int i = 0; i < dZ.GetLength(0); i++)
            {
                for (int j = 0; j < dZ.GetLength(1); j++)
                {
                    dBiases[j] += dZ[i, j];
                }
            }
            //Gradients on regularization
            if (L1W > 0)
            {
                for (int i = 0; i < Weights.GetLength(0); i++)
                {
                    for (int j = 0; j < Weights.GetLength(1); j++)
                    {
                        dWeights[i, j] += L1W * Math.Sign(Weights[i, j]);
                    }
                }
            }
            if (L2W > 0)
            {
                for (int i = 0; i < Weights.GetLength(0); i++)
                {
                    for (int j = 0; j < Weights.GetLength(1); j++)
                    {
                        dWeights[i, j] += L2W * 2 * Weights[i, j];
                    }
                }
            }
            if (L1B > 0)
            {
                for (int i = 0; i < Biases.Length; i++)
                {
                    dBiases[i] += L1B * Math.Sign(Biases[i]);
                }
            }
            if (L2B > 0)
            {
                for (int i = 0; i < Biases.Length; i++)
                {
                    dBiases[i] += L2B * 2 * Biases[i];
                }
            }

            //Gradients on values
            Dinputs = AdditionalMath.Matrix_Multiplier(dZ, AdditionalMath.Transpose(Weights));
        }


        public void ZeroGrad()
        {
            Array.Clear(dWeights, 0, dWeights.Length);

            Array.Clear(dBiases, 0, dBiases.Length);
        }
    }
}
