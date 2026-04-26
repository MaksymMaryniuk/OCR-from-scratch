using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace Model
{
    public static class AdditionalMath
    {
        private static Random rand = new Random();
        public static void Matrix_filler(float[,] value)
        {

            for (int i = 0; i < value.GetLength(0); i++)
            {
                for (int j = 0; j < value.GetLength(1); j++)
                {
                    value[i, j] = (float)((rand.NextDouble() * 2 - 1) * 0.1);
                }
            }
        }
        public static void Vector_filler(float[] value)
        {
            for (int i = 0; i < value.Length; i++)
            {
                value[i] = (float)(rand.NextDouble() * 2 - 1);
            }
        }

        public static float[,] Matrix_Multiplier(float[,] matrix1, float[,] matrix2)
        {
            int rows = matrix1.GetLength(0);
            int cols = matrix2.GetLength(1);
            int common = matrix1.GetLength(1);

            float[,] output = new float[rows, cols];

            Parallel.For(0, rows, i =>
            {
                for (int k = 0; k < common; k++)
                {
                    float a = matrix1[i, k];
                    for (int j = 0; j < cols; j++)
                    {
                        output[i, j] += a * matrix2[k, j];
                    }
                }
            });

            return output;
        }

        public static float[,] Transpose(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            float[,] transposed = new float[cols, rows];

            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    transposed[j, i] = matrix[i, j];
                }
            });
            return transposed;
        }
        public static float SumOfAbsoluteValues(float[,] matrix)
        {
            float sum = 0;
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    sum += Math.Abs(matrix[i, j]);
                }
            }
            return sum;
        }
        public static float SumOfAbsoluteValues(float[] vector)
        {
            float sum = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                sum += Math.Abs(vector[i]);
            }
            return sum;
        }
        public static float SumOfSquares(float[,] matrix)
        {
            float sum = 0;
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    sum += matrix[i, j] * matrix[i, j];
                }
            }
            return sum;
        }
        public static float SumOfSquares(float[] vector)
        {
            float sum = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                sum += vector[i] * vector[i];
            }
            return sum;
        }
        public static int GetArgmax(float[,] predictions)
        {
            int numClasses = predictions.GetLength(1);
            int bestClass = 0;
            float maxVal = predictions[0, 0];

            for (int i = 1; i < numClasses; i++)
            {
                if (predictions[0, i] > maxVal)
                {
                    maxVal = predictions[0, i];
                    bestClass = i;
                }
            }

            return bestClass;
        }
    }
}
