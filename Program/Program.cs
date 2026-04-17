using Model;
internal class Program
{
    private static void Main(string[] args)
    {
        // Model training example with 100 samples, 3 features, and 3 classes for classification (Random values)
        int rows = 100;
        int cols = 5;
        double min = -10.0;
        double max = 100.0;
        double[,] matrix = new double[rows, cols];

        Random rand = new Random();


        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double val = min + (rand.NextDouble() * (max - min));
                matrix[i, j] = Math.Round(val, 2);
            }
        }

        (double[,] X, int[] y) = DataGenerator.CreateData(rows, cols);

        int epochs = 10001;


        Model.Dense layer1 = new Model.Dense(X.GetLength(1), 64);
        Model.Dense layer2 = new Model.Dense(64, 5);
        Model.ActivationReLU relu = new Model.ActivationReLU();
        Model.ActivationSoftmax softmax = new Model.ActivationSoftmax();
        Model.LossCCE loss = new Model.LossCCE();

        // Optimizers
        Model.Optimizer_SGD sgdOptim = new Model.Optimizer_SGD(0.001, 0.0001, 0.9);
        Model.Optimizer_AdaGrad rmsOptimadaOptim = new Model.Optimizer_AdaGrad(1.0, 1e-5);
        Model.Optimizer_RMSProp rmsOptim = new Model.Optimizer_RMSProp(0.9, 0.01, 1e-5);
        Model.Optimizer_Adam adamOptim = new Model.Optimizer_Adam(0.9, 0.999, 0.02, 0.00001);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // 1. Forward
            layer1.Forward(X);
            relu.Forward(layer1.Output);
            layer2.Forward(relu.Output);
            softmax.Forward(layer2.Output);

            // 2. Loss
            double lossValue = loss.Calculate(softmax.Output, y);

            // 3. Backward
            double[,] dZ = softmax.Backward(y);

            dZ = layer2.Backward(dZ);
            dZ = relu.Backward(dZ);
            dZ = layer1.Backward(dZ);

            // 4. Update
            adamOptim.PreUpdate();
            adamOptim.Update(layer1);
            adamOptim.Update(layer2);
            adamOptim.PostUpdate();

            layer1.ZeroGrad();
            layer2.ZeroGrad();

            // logs
            if (epoch % 1000 == 0)
            {
                Console.WriteLine("------------------------------");
                Console.WriteLine($"Epoch {epoch}, Loss: {lossValue}");
                Console.WriteLine($"Accuracy: {PrintAccuracyInt(softmax.Output, y) * 100}%");
                Console.WriteLine($"Lr: {adamOptim.currentLearningRate}");
                Console.WriteLine("------------------------------");
            }
        }

        void PrintMatrix(double[,] matrix)
        {

            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    Console.Write(Math.Round(matrix[i, j], 2) + "\t");
                }
                Console.WriteLine();
            }

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
    }
}
