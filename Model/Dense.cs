using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public class Dense : Layer
    {


        public double[,] Weights { get; set; }
        public double[] Biases { get; set; }
        public double[,] dWeights { get; set; }
        public double[] dBiases { get; set; }
        public double[,] WeightMomentums { get; set; }
        public double[] BiasMomentums { get; set; }
        public double[,] WeightCache { get; set; }
        public double[] BiasCache { get; set; }

        private Random rand = new Random();
        public Dense(int num_inputs, int num_neurons)
        {
            Weights = new double[num_inputs, num_neurons];
            Biases = new double[num_neurons];
            Output = new double[num_inputs, num_neurons];
            WeightMomentums = new double[num_inputs, num_neurons];
            BiasMomentums = new double[num_neurons];
            WeightCache = new double[num_inputs, num_neurons];
            BiasCache = new double[num_neurons];
            Matrix_filler(Weights);
            Vector_filler(Biases);
        }
        public override void Forward(double[,] X)
        {
            Inputs = X;
            Output = Matrix_Multiplier(X, Weights);

            for (int i = 0; i < Output.GetLength(0); i++)
            {
                for (int j = 0; j < Output.GetLength(1); j++)
                {
                    Output[i, j] += Biases[j];
                }
            }
        }
        public override double[,] Backward(double[,] dZ)
        {
            /*operations: dW = X^T * dZ
                       db = sum(dZ)
                       dX = dZ * W^T
             where X - inputs, dZ - grad due to output layer, W - weights of layer,
             dW - grad due to weights,
             db - grad due to biases,
             dX - grad due to inputs
            */

            dWeights = Matrix_Multiplier(Transpose(Inputs), dZ);
            dBiases = new double[dZ.GetLength(1)];

            for (int i = 0; i < dZ.GetLength(0); i++)
            {
                for (int j = 0; j < dZ.GetLength(1); j++)
                {
                    dBiases[j] += dZ[i, j];
                }
            }

            double[,] dX = Matrix_Multiplier(dZ, Transpose(Weights));
            return dX;
        }

        private void Matrix_filler(double[,] value)
        {

            for (int i = 0; i < value.GetLength(0); i++)
            {
                for (int j = 0; j < value.GetLength(1); j++)
                {
                    value[i, j] = (rand.NextDouble() * 2 - 1) * 0.1;
                }
            }
        }
        private void Vector_filler(double[] value)
        {
                for (int i = 0; i < value.Length; i++)
                {
                    value[i] = rand.NextDouble() * 2 - 1;
                }
        }

        private double[,] Matrix_Multiplier(double[,] matrix1, double[,] matrix2)
        {
            int output_rows = matrix1.GetLength(0);
            int output_cols = matrix2.GetLength(1);
            double[,] output = new double[output_rows, output_cols];

            for (int rows = 0; rows < output_rows; rows++)
            {
                for (int cols = 0; cols < output_cols; cols++)
                {
                    double sum = 0;
                    for (int common = 0; common < matrix1.GetLength(1); common++)
                    {
                        sum += matrix1[rows, common] * matrix2[common, cols];
                    }
                    output[rows, cols] = sum;
                }
            }
            return output;
        }

        private double[,] Transpose(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[,] transposed = new double[cols, rows];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    transposed[j, i] = matrix[i, j];
                }
            }
            return transposed;
        }
        private void ZeroFiller(double[,] matrix)
        {
            int cols = matrix.GetLength(0);
            int rows = matrix.GetLength(1);
            for (int i = 0; i < cols; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    matrix[i, j] = 0;
                }
            }
        }
        public void ZeroGrad()
        {
            // Очищаємо масив градієнтів ваг
            Array.Clear(dWeights, 0, dWeights.Length);

            // Очищаємо масив градієнтів зсувів
            Array.Clear(dBiases, 0, dBiases.Length);
        }
    }
}
