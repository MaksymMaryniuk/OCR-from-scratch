using ScottPlot;
using Model;
internal class Program
{
    private static void Main(string[] args)
    {
        int rows = 2;
        int cols = 4;
        double min = 4.0;
        double max = 7.0;
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
        Console.WriteLine("------Basic Inputs (X)------");
        PrintMatrix(matrix);
        Console.WriteLine("----------------------------");

        Model.Layer layer1 = new Model.Layer(4, 5);
        Model.Layer layer2 = new Model.Layer(5, 1);

        Console.WriteLine("------Weights------");
        PrintMatrix(layer1.Weights);
        Console.WriteLine("----------------------------");

        Console.WriteLine("------Biases------");
        for (int i = 0; i < layer1.Biases.Length; i++)
        {
            Console.Write(Math.Round(layer1.Biases[i], 2) + "\t");
        }
        Console.WriteLine("\n----------------------------");

        layer1.Forward(matrix);
        Console.WriteLine("------After forward------");
        PrintMatrix(layer1.Output);
        Console.WriteLine("----------------------------");
        layer2.Forward(layer1.Output);
        Console.WriteLine("------After forward 2------");
        PrintMatrix(layer2.Output);
        Console.WriteLine("----------------------------");
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
        int pointsPerClass = 100;
        int numClasses = 3;
        var (X, y) = DataGenerator.CreateData(pointsPerClass, numClasses);
        double[] xs = new double[pointsPerClass];
        double[] ys = new double[pointsPerClass];


        var plt = new ScottPlot.Plot();

        plt.Title("Spiral Data (C# + ScottPlot)");

        for (int c = 0; c < numClasses; c++)
        {
            for (int i = 0; i < pointsPerClass; i++)
            {
                int idx = c * pointsPerClass + i;
                xs[i] = X[idx, 0];
                ys[i] = X[idx, 1];
            }
            var scatter = plt.Add.Scatter(xs, ys);
            scatter.LineWidth = 0; 
            scatter.MarkerSize = 5;
            Console.WriteLine(System.IO.Path.GetFullPath($"spiral_plot{c}.png"));
            plt.SavePng($"spiral_plot{c}.png", 600, 600);
        }

        plt.ShowLegend();

    }
}
