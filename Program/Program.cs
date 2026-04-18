using Model;
using Model.Layers;
using Model.Optimizers;
internal class Program
{
    private static void Main(string[] args)
    {
        // Model training example with 100 samples, 3 features, and 3 classes for classification (Random values)
        int rows = 100;
        int cols = 3;

        (double[,] X, int[] y) = DataGenerator.CreateData(rows, cols);

        int epochs = 10000;

        // Layers
        Layer_Dense layer1 = new Model.Layers.Layer_Dense(X.GetLength(1), 256, 0, 0.000001, 0, 0.000001);
        Layer_Dense layer2 = new Model.Layers.Layer_Dense(256, 3);
        Layer_Dropout dropout = new Model.Layers.Layer_Dropout(0.1);
        ActivationReLU relu = new Model.Layers.ActivationReLU();
        ActivationSoftmax softmax = new Model.Layers.ActivationSoftmax();
        // Loss
        Model.LossCCE loss = new Model.LossCCE();

        // Optimizer
        Optimizer_Adam adamOptim = new Model.Optimizers.Optimizer_Adam(0.9, 0.999, 0.02, 0.00001);


        // Model
        Model.Model model = new Model.Model();

        model.Add(layer1);
        model.Add(relu);
        model.Add(dropout);
        model.Add(layer2);
        model.Add(softmax);

        model.set(optimizer: adamOptim, loss: loss);

        model.Train(X, y, epochs);

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
    }
}
