using Model.Layers;
using Model.Optimizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public class Model
    {
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

        public void Train(double[,] X, int[] y, int epochs = 1000)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Forward pass
                double[,] output = Forward(X);
                // Compute loss
                double lossValue = Loss.Calculate(output, y);
                // Backward pass
                double[,] dOutput = Loss.Backward(output, y);

                Backward(dOutput);

                // Update weights and biases
                Optimizer.PreUpdate();
                foreach (var layer in Layers)
                {
                    if (layer is Layer_Dense denseLayer)
                    {
                        Optimizer.Update(denseLayer);
                    }
                }
                Optimizer.PostUpdate();
                // Optionally print loss every certain number of epochs
                if (epoch % 100 == 0)
                {
                    Console.WriteLine($"Epoch {epoch}, Loss: {lossValue}, Accuracy: {PrintAccuracyInt(output, y) * 100}%");
                }
            }
        }

        /*public void Finalize()
        {
            InputLayer = new Layer_Input();

            for (int i = 0; i < Layers.Count; i++)
            {
                if (i == 0)
                {
                    Layers[i].Prev = InputLayer;
                    Layers[i].Next = Layers[i + 1];
                }
                else if (i < Layers.Count - 1)
                {
                    Layers[i].Prev = Layers[i - 1];
                    Layers[i].Next = Layers[i + 1];
                }
                else
                {
                    Layers[i].Prev = Layers[i - 1];
                    Layers[i].Next = Loss;
                }
            }
        }*/
        public double[,] Forward(double[,] X)
        {
            double[,] output = X;
            foreach (var layer in Layers)
            {
                layer.Forward(output);
                output = layer.Output;
            }
            return output;
        }
        public void Backward(double[,] dvalues)
        {
            for (int i = Layers.Count - 2; i >= 0; i--)
            {
                Layers[i].Backward(dvalues);
                dvalues = Layers[i].Dinputs;
            }
        }

        double PrintAccuracyInt(double[,] softmax_output, int[] target)
        {
            int rows = softmax_output.GetLength(0);
            int cols = softmax_output.GetLength(1);
            double correct = 0;
            for (int i = 0; i < rows; i++)
            {
                double maxVal = double.MinValue;
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
        double PrintAccuracy(double[,] softmax_output, double[,] target)
        {
            int rows = softmax_output.GetLength(0);
            int cols = softmax_output.GetLength(1);
            double correct = 0;

            for (int i = 0; i < rows; i++)
            {
                double maxVal = double.MinValue;
                int predictedIndex = -1;
                for (int j = 0; j < cols; j++)
                {
                    if (softmax_output[i, j] > maxVal)
                    {
                        maxVal = softmax_output[i, j];
                        predictedIndex = j;
                    }
                }
                if (target[i, predictedIndex] == 1.0)
                {
                    correct++;
                }
            }

            return correct / rows;
        }
    }
}
