using System;
using System.Collections.Generic;
using System.Text;

namespace Model.Layers
{
    public class Layer_Dense : Layer
    {
        public float[,] Weights { get; set; }
        public float[] Biases { get; set; }
        public float[,] dWeights { get; set; }
        public float[] dBiases { get; set; }
        public float[,] WeightMomentums { get; set; }
        public float[] BiasMomentums { get; set; }
        public float[,] WeightCache { get; set; }
        public float[] BiasCache { get; set; }
        public float L1W { get; set; }
        public float L2W { get; set; }
        public float L1B { get; set; }
        public float L2B { get; set; }


        public Layer_Dense(int num_inputs, int num_neurons, float l1w = 0.0F, float l2w = 0.0F, float l1b = 0.0F, float l2b = 0.0F)
        {
            Weights = new float[num_inputs, num_neurons];
            Biases = new float[num_neurons];

            WeightMomentums = new float[num_inputs, num_neurons];
            BiasMomentums = new float[num_neurons];
            WeightCache = new float[num_inputs, num_neurons];
            BiasCache = new float[num_neurons];
            L1W = l1w;
            L2W = l2w;
            L1B = l1b;
            L2B = l2b;
            AdditionalMath.Matrix_filler(Weights);
            AdditionalMath.Vector_filler(Biases);
        }
        public override void Forward(float[,] X)
        {
            Output = new float[X.GetLength(0), Weights.GetLength(1)];
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

        public override void Backward(float[,] dZ)
        {
            int batchSize = dZ.GetLength(0);
            int inputCount = Inputs.GetLength(1);
            int neuronCount = dZ.GetLength(1);

            // 1. ЛІНИВА ІНІЦІАЛІЗАЦІЯ: створюємо масиви тільки якщо розмір змінився або їх немає
            if (dWeights == null || dWeights.GetLength(0) != inputCount || dWeights.GetLength(1) != neuronCount)
                dWeights = new float[inputCount, neuronCount];

            if (dBiases == null || dBiases.Length != neuronCount)
                dBiases = new float[neuronCount];

            if (Dinputs == null || Dinputs.GetLength(0) != batchSize || Dinputs.GetLength(1) != inputCount)
                Dinputs = new float[batchSize, inputCount];

            // 2. ГРАДІЄНТ ПО ВАГАХ (dW = X^T * dZ)
            // Паралелимо по входах (inputCount)
            System.Threading.Tasks.Parallel.For(0, inputCount, i =>
            {
                for (int j = 0; j < neuronCount; j++)
                {
                    float sum = 0;
                    for (int s = 0; s < batchSize; s++)
                    {
                        sum += Inputs[s, i] * dZ[s, j];
                    }
                    dWeights[i, j] = sum / batchSize;
                }
            });

            // 3. ГРАДІЄНТ ПО БІАСАХ (db = sum(dZ))
            System.Threading.Tasks.Parallel.For(0, neuronCount, j =>
            {
                float sum = 0;
                for (int s = 0; s < batchSize; s++)
                {
                    sum += dZ[s, j];
                }
                dBiases[j] = sum / batchSize;
            });

            // 4. ГРАДІЄНТ ПО ВХОДАХ (dX = dZ * W^T)
            // Оптимізація: читаємо Weights[i, j] послідовно в межах рядка
            System.Threading.Tasks.Parallel.For(0, batchSize, s =>
            {
                for (int i = 0; i < inputCount; i++)
                {
                    float sum = 0;
                    for (int j = 0; j < neuronCount; j++)
                    {
                        // Тут j змінюється найшвидше - це краще для кешу
                        sum += dZ[s, j] * Weights[i, j];
                    }
                    Dinputs[s, i] = sum;
                }
            });
        }


        public void ZeroGrad()
        {
            if (Weights == null || Biases == null)
                return;

            if (dWeights == null)
                dWeights = new float[Weights.GetLength(0), Weights.GetLength(1)];
            else
                Array.Clear(dWeights, 0, dWeights.Length);

            if (dBiases == null)
                dBiases = new float[Biases.Length];
            else
                Array.Clear(dBiases, 0, dBiases.Length);
        }
    }
}
