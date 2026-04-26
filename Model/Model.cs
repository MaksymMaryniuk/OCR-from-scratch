using Model.Layers;
using Model.Optimizers;
using System;
using System.Collections.Generic;
using System.Data.Common;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Model
{
    public class Model
    {
        private float[,] _xBatchBuffer;
        private int[] _yBatchBuffer;
        public float AverageAccuracy { get; set; }

        public List<Layer> Layers { get; set; }
        public Optimizer Optimizer { get; set; }
        public Loss Loss { get; set; }

        public Model()
        {
            Layers = new List<Layer>();
        }
        public Model(Layer[] layers)
        {
            Layers = new List<Layer>(layers);
        }

        public void Add(Layer layer)
        {
            Layers.Add(layer);
        }

        public void set(Optimizer optimizer, Loss loss)
        {
            Optimizer = optimizer;
            Loss = loss;
        }

        public void Train(float[,] X, int[] y, int epochs = 1000, int batchSize = 32)
        {
            int samples = X.GetLength(0);
            int features = X.GetLength(1);
            int[] indices = Enumerable.Range(0, samples).ToArray();
            float epochLoss = 0;
            float epochAccuracy = 0;
            int batches = 0;
            Random rng = new Random();

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                Shuffle(indices, rng);

                epochLoss = 0;
                epochAccuracy = 0;
                batches = 0;

                for (int i = 0; i < samples; i += batchSize)
                {
                    int end = Math.Min(i + batchSize, samples);
                    int currentBatchSize = end - i;

                    float[,] X_batch = GetBatch(X, indices, i, currentBatchSize, features);
                    int[] y_batch = GetBatchLabels(y, indices, i, currentBatchSize);

                    // --- Forward ---
                    float[,] output = Forward(X_batch);


                    foreach (var layer in Layers)
                    {
                        if (layer is Layer_Dense denseLayer)
                            denseLayer.ZeroGrad();
                    }

                    // --- Accuracy for batch ---
                    float batchAccuracy = PrintAccuracy(output, y_batch);
                    epochAccuracy += batchAccuracy;

                    // --- Loss ---
                    float lossValue = Loss.Calculate(output, y_batch);
                    epochLoss += lossValue;
                    batches++;

                    // --- Backward & Update ---
                    float[,] dOutput = Loss.Backward(output, y_batch);
                    Backward(dOutput);

                    Optimizer.PreUpdate();
                    foreach (var layer in Layers)
                    {
                        if (layer is Layer_Dense denseLayer)
                            Optimizer.Update(denseLayer);
                    }
                    Optimizer.PostUpdate();
                }

                AverageAccuracy = epochAccuracy / batches;

                if (epoch % 5 == 0)
                {
                    float averageLoss = epochLoss / batches;
                    Console.WriteLine($"Epoch {epoch}: Loss = {averageLoss:F4}, Accuracy = {AverageAccuracy:F4}");
                }
            }
        }

        public void TrainEMNIST(float[,] X, int[] y, int epochs = 10, int batchSize = 64)
        {
            int samples = X.GetLength(0);
            int[] indices = Enumerable.Range(0, samples).ToArray();
            Random rng = new Random();

            _xBatchBuffer = new float[batchSize, X.GetLength(1)];
            _yBatchBuffer = new int[batchSize];

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                Shuffle(indices, rng);
                float totalLoss = 0;
                int batchCount = 0;
                float epochAccuracy = 0;

                for (int i = 0; i < samples; i += batchSize)
                {
                    int currentBatchSize = Math.Min(batchSize, samples - i);


                    PrepareBatch(X, indices, i, currentBatchSize, X.GetLength(1), y);

                    // --- Forward ---
                    float[,] output = Forward(_xBatchBuffer);

                    // --- Accuracy for batch ---
                    float batchAccuracy = PrintAccuracy(output, _yBatchBuffer);
                    epochAccuracy += batchAccuracy;

                    totalLoss += Loss.Calculate(output, _yBatchBuffer);



                    // --- Backward & Update ---
                    foreach (var layer in Layers) if (layer is Layer_Dense d) d.ZeroGrad();

                    float[,] dOutput = Loss.Backward(output, _yBatchBuffer);
                    Backward(dOutput);


                    Optimizer.PreUpdate();
                    foreach (var layer in Layers) if (layer is Layer_Dense d) Optimizer.Update(d);
                    Optimizer.PostUpdate();

                    batchCount++;
                }
                AverageAccuracy = totalLoss / batchCount;
                Console.WriteLine($"Епоха {epoch}/{epochs} - Loss: {totalLoss / batchCount:F4}");
                Console.WriteLine($"Точність після епохи {epoch}: {AverageAccuracy:F4}");
            }
        }

        private void PrepareBatch(float[,] X, int[] indices, int start, int batchSize, int features, int[] y)
        {
            Parallel.For(0, batchSize, i =>
            {
                int originalIndex = indices[start + i];
                for (int j = 0; j < features; j++)
                {
                    _xBatchBuffer[i, j] = X[originalIndex, j];
                }
                _yBatchBuffer[i] = y[originalIndex];
            });
        }


        private void Shuffle(int[] array, Random rng)
        {
            int n = array.Length;
            while (n > 1)
            {
                int k = rng.Next(n--);
                int temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }
        private float[,] GetBatch(float[,] X, int[] indices, int start, int batchSize, int features)
        {
            float[,] batch = new float[batchSize, features];

            for (int i = 0; i < batchSize; i++)
            {
                int originalIndex = indices[start + i];

                for (int j = 0; j < features; j++)
                {
                    batch[i, j] = X[originalIndex, j];
                }
            }

            return batch;
        }
        private int[] GetBatchLabels(int[] y, int[] indices, int start, int batchSize)
        {
            int[] batch = new int[batchSize];

            for (int i = 0; i < batchSize; i++)
            {
                int originalIndex = indices[start + i];
                batch[i] = y[originalIndex];
            }

            return batch;
        }
        public float[,] Forward(float[,] X)
        {
            float[,] output = X;
            foreach (var layer in Layers)
            {
                layer.Forward(output);
                output = layer.Output;
            }
            return output;
        }
        public void Backward(float[,] dvalues)
        {
            for (int i = Layers.Count - 2; i >= 0; i--)
            {
                Layers[i].Backward(dvalues);
                dvalues = Layers[i].Dinputs;
            }
        }

        float PrintAccuracy(float[,] softmax_output, int[] target)
        {
            int rows = softmax_output.GetLength(0);
            int cols = softmax_output.GetLength(1);
            float correct = 0;
            for (int i = 0; i < rows; i++)
            {
                float maxVal = float.MinValue;
                int predictedIndex = -1;
                for (int j = 0; j < cols; j++)
                {
                    if (softmax_output[i, j] > maxVal)
                    {
                        maxVal = softmax_output[i, j];
                        predictedIndex = j;
                    }
                }
                if (target[i] == predictedIndex)
                {
                    correct++;
                }
            }
            return correct / rows;
        }

        public static (float[,], int[]) LoadEMNIST(string filePath, int maxSamples = 112800)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("file EMNIST not found!");

            var allLines = File.ReadAllLines(filePath);
            int samples = Math.Min(allLines.Length, maxSamples);

            float[,] X = new float[samples, 784];
            int[] y = new int[samples];

            for (int i = 0; i < samples; i++)
            {
                string[] values = allLines[i].Split(',');
                y[i] = int.Parse(values[0]);

                for (int r = 0; r < 28; r++)
                {
                    for (int c = 0; c < 28; c++)
                    {
                        X[i, r * 28 + c] = (float)(double.Parse(values[c * 28 + r + 1]) / 255.0);
                    }
                }
            }
            return (X, y);
        }


    }
}
