using Model;
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
        Model.Layer_Dense layer1 = new Model.Layer_Dense(X.GetLength(1), 256, 0, 0.000001, 0, 0.000001);
        Model.Layer_Dense layer2 = new Model.Layer_Dense(256, 3);
        Model.Layer_Dropout dropout = new Model.Layer_Dropout(0.1);
        Model.ActivationReLU relu = new Model.ActivationReLU();
        Model.ActivationSoftmax softmax = new Model.ActivationSoftmax();

        // Loss
        Model.LossCCE loss = new Model.LossCCE();

        // Optimizer
        Model.Optimizer_Adam adamOptim = new Model.Optimizer_Adam(0.9, 0.999, 0.02, 0.00001);


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
