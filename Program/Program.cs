using Model;
using Model.Layers;
using Model.Optimizers;
using SkiaSharp;
using System.Drawing;
internal class Program
{
    private static void Main(string[] args)
    {
        string projectRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\..\"));

        var (X, y) = Model.Model.LoadEMNIST(Path.Combine(projectRoot, "Model", "data", "emnist-balanced-train.csv"));

        int epochs = 50;

        // Layers
        Layer_Dense layer1 = new Model.Layers.Layer_Dense(784, 256);
        Layer_Dense layer2 = new Model.Layers.Layer_Dense(256, 128);
        Layer_Dense layer3 = new Model.Layers.Layer_Dense(128, 47);
        ActivationReLU relu1 = new Model.Layers.ActivationReLU();
        ActivationReLU relu2 = new Model.Layers.ActivationReLU();
        ActivationSoftmax softmax = new Model.Layers.ActivationSoftmax();
        // Loss
        Model.LossCCE loss = new Model.LossCCE();
        
        // Optimizer
        Optimizer_Adam adamOptim = new Model.Optimizers.Optimizer_Adam(0.9F, 0.999F, 0.001F, 0.00001F);


        // Model
        Model.Model model = new Model.Model();

        model.Add(layer1);
        model.Add(relu1);
        model.Add(layer2);
        model.Add(relu2);
        model.Add(layer3);
        model.Add(softmax);

        model.set(optimizer: adamOptim, loss: loss);

        model.TrainEMNIST(X, y, epochs, 128);
        Model.ModelSaver.SaveJson(Path.Combine(projectRoot,"Optical Recognition Pattern", "model_config2.json"), model.Layers);

    }


}
