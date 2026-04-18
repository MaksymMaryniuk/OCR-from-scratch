using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public static class AdditionalMath
    {
        private static Random rand = new Random();
        public static void Matrix_filler(double[,] value)
        {

            for (int i = 0; i < value.GetLength(0); i++)
            {
                for (int j = 0; j < value.GetLength(1); j++)
                {
                    value[i, j] = (rand.NextDouble() * 2 - 1) * 0.1;
                }
            }
        }
        public static void Vector_filler(double[] value)
        {
            for (int i = 0; i < value.Length; i++)
            {
                value[i] = rand.NextDouble() * 2 - 1;
            }
        }

        public static double[,] Matrix_Multiplier(double[,] matrix1, double[,] matrix2)
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

        public static double[,] Transpose(double[,] matrix)
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
        public static double SumOfAbsoluteValues(double[,] matrix)
        {
            double sum = 0;
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    sum += Math.Abs(matrix[i, j]);
                }
            }
            return sum;
        }
        public static double SumOfAbsoluteValues(double[] vector)
        {
            double sum = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                sum += Math.Abs(vector[i]);
            }
            return sum;
        }
        public static double SumOfSquares(double[,] matrix)
        {
            double sum = 0;
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    sum += matrix[i, j] * matrix[i, j];
                }
            }
            return sum;
        }
        public static double SumOfSquares(double[] vector)
        {
            double sum = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                sum += vector[i] * vector[i];
            }
            return sum;
        }
    }
}
