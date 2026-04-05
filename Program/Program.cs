using ScottPlot;
using Model;
internal class Program
{
    private static void Main(string[] args)
    {
        // Model training example with 2 samples, 3 features, and 3 classes for classification (Random values)
        int rows = 2;
        int cols = 3;
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

        
        int epochs = 1000;

        double[,] y_true = { { 0, 1, 0 }, { 1, 0, 0 } };


        Model.Layer layer1 = new Model.Layer(cols, 5);
        Model.Layer layer2 = new Model.Layer(5, 3);
        Model.ActivationReLU relu = new Model.ActivationReLU();
        Model.ActivationSoftmax softmax = new Model.ActivationSoftmax(); 
        Model.LossCCE loss = new Model.LossCCE();
        Model.Optimizer_SGD optimizer = new Model.Optimizer_SGD();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // 1. Forward
            layer1.Forward(matrix);
            relu.Forward(layer1.Output);
            layer2.Forward(relu.Output);
            softmax.Forward(layer2.Output);

            // 2. Loss
            double lossValue = loss.Calculate(softmax.Output, y_true);

            // 3. Backward
            double[,] dZ = loss.Backward(softmax.Output, y_true);

            dZ = layer2.Backward(dZ);
            dZ = relu.Backward(dZ);
            dZ = layer1.Backward(dZ);

            // 4. Update
            optimizer.Update(layer1);
            optimizer.Update(layer2);

            // logs
            if (epoch % 100 == 0)
            {
                Console.WriteLine($"Epoch {epoch}, Loss: {lossValue}");
                Console.WriteLine($"Accuracy: {PrintAccuracy(softmax.Output, y_true) * 100}%");
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
    }
}
