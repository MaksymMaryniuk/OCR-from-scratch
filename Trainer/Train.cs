using Vionet;
using Vionet.Data;
using Vionet.Layers;
using Vionet.Optimizers;
using System.Drawing;
using System.Reflection;
internal class Program
{
    static string projectRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\..\"));
    private static void Main(string[] args)
    {
        //ProceedGeneratedDataTrain();
        ProceedEMNISTTrain();
    }



    public static void ProceedGeneratedDataTrain()
    {
        string labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;()\"'";
        string labelsCyrrilic= "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиіїйклмнопрстуфхцчшщьюя0123456789.,!?:;()-\"'+= ";
        string[] fonts = { "Arial", "Times New Roman", "Verdana", "Courier New", "Calibri", "Tahoma", "Georgia" };


        var config = new AugmentationConfig
        {
            UseBlur = true,
            BlurChance = 0.2,
            UseSaltPepper = true,
            SaltPepperIntensity = 0.01,
            UseBrightness = true,
            UseShift = true,
            MaxShift = 3
        };

        var (X, y) = DatasetGenerator.GeneratePrintedData(fonts, labelsCyrrilic, config, 50);


        int total = X.GetLength(0);
        Console.WriteLine($"Generated {total} printed character samples.");

        int[] indices = Enumerable.Range(0, total).ToArray();
        Random rng = new Random(34);
        indices = indices.OrderBy(_ => rng.Next()).ToArray();

        float[,] X_shuffled = new float[total, 784];

        int[] y_shuffled = new int[total];
        for (int i = 0; i < total; i++)
        {
            y_shuffled[i] = y[indices[i]];
            for (int j = 0; j < 784; j++)
                X_shuffled[i, j] = X[indices[i], j];
        }

        for (int i = 0; i < 20; i++)
        {
            ImageHelper.SaveSampleFrom2DArray(X_shuffled, i, 28, 28, Path.Combine(projectRoot, "Program", "Logs-Printed", $"test_sample{i + 21}.png"));
        }

        // Layers
        Layer_Dense layer1 = new Vionet.Layers.Layer_Dense(784, 256);
        Layer_Dense layer2 = new Vionet.Layers.Layer_Dense(256, 128);
        Layer_Dense layer3 = new Vionet.Layers.Layer_Dense(128, 64);
        Layer_Dense layer4 = new Vionet.Layers.Layer_Dense(64, labelsCyrrilic.Length);
        Layer_Dropout dropout1 = new Vionet.Layers.Layer_Dropout(0.3F);
        Layer_Dropout dropout2 = new Vionet.Layers.Layer_Dropout(0.2F);
        ActivationReLU relu1 = new Vionet.Layers.ActivationReLU();
        ActivationReLU relu2 = new Vionet.Layers.ActivationReLU();
        ActivationReLU relu3 = new Vionet.Layers.ActivationReLU();
        ActivationSoftmax softmax = new Vionet.Layers.ActivationSoftmax();

        // Loss
        Vionet.LossCCE loss = new Vionet.LossCCE();
        // Optimizer
        Vionet.Optimizers.Optimizer_Adam adamOptim = new Vionet.Optimizers.Optimizer_Adam(0.9F, 0.999F, 0.001F, 0.00001F);

        // Model
        Vionet.Model model = new Vionet.Model();

        model.Add(layer1); model.Add(relu1); model.Add(dropout1);
        model.Add(layer2); model.Add(relu2); model.Add(dropout2);
        model.Add(layer3); model.Add(relu3);
        model.Add(layer4); model.Add(softmax);

        model.set(optimizer: adamOptim, loss: loss);

        // Split
        var (X_train, X_val, y_train, y_val) = ModelSelection.TrainValSplit(X_shuffled, y_shuffled, valSplit: 0.2f);

        // Train
        model.TrainEMNIST(X_train, y_train, epochs: 15, batchSize: 128);

        // Validate
        double acc = ModelSelection.Validate(model, X_val, y_val, labels: labelsCyrrilic, weakThreshold: 80.0);

        // Save model
        Vionet.ModelSaver.SaveJson(Path.Combine(projectRoot, "OCR", "NeuralNetworks", "model_config_printedCyrrilic.json"), model.Layers);
    }



    public static void ProceedEMNISTTrain()
    {
        string emnistLabels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";

        // Load EMNIST data
        var (raw_X, y) = Vionet.Model.LoadEMNIST(Path.Combine(projectRoot, "Model", "Data", "emnist-balanced-train.csv"));

        // Data augmentation config
        var config = new AugmentationConfig
        {
            UseBlur = true,
            BlurChance = 0.2,
            UseSaltPepper = true,
            SaltPepperIntensity = 0.01,
            UseBrightness = true,
            UseShift = true,
            MaxShift = 3
        };

        float[,] noisyX = DataAugmentation.AugmentExistingDataset(raw_X, config);

        int total = noisyX.GetLength(0);
        Console.WriteLine($"Took {total} EMNIST character samples.");

        int[] indices = Enumerable.Range(0, total).ToArray();
        Random rng = new Random(42);
        indices = indices.OrderBy(_ => rng.Next()).ToArray();

        float[,] X_shuffled = new float[total, 784];

        int[] y_shuffled = new int[total];
        for (int i = 0; i < total; i++)
        {
            y_shuffled[i] = y[indices[i]];
            for (int j = 0; j < 784; j++)
                X_shuffled[i, j] = noisyX[indices[i], j];
        }

        for (int i = 0; i < 100; i++)
        {
            ImageHelper.SaveSampleFrom2DArray(X_shuffled, i, 28, 28, Path.Combine(projectRoot, "Program", "Logs-EMNIST", $"test_sample{i + 1}.png"));
        }

        // Layers
        Vionet.Layers.Layer_Dense layer1 = new Vionet.Layers.Layer_Dense(784, 256);
        Vionet.Layers.Layer_Dense layer2 = new Vionet.Layers.Layer_Dense(256, 128);
        //Vionet.Layers.Layer_Dense layer3 = new Vionet.Layers.Layer_Dense(128, 64);
        Vionet.Layers.Layer_Dense layer4 = new Vionet.Layers.Layer_Dense(128, emnistLabels.Length);
        Vionet.Layers.Layer_Dropout dropout1 = new Vionet.Layers.Layer_Dropout(0.3F);
        Vionet.Layers.Layer_Dropout dropout2 = new Vionet.Layers.Layer_Dropout(0.2F);
        Vionet.Layers.ActivationReLU relu1 = new Vionet.Layers.ActivationReLU();
        Vionet.Layers.ActivationReLU relu2 = new Vionet.Layers.ActivationReLU();
        //Vionet.Layers.ActivationReLU relu3 = new Vionet.Layers.ActivationReLU();
        Vionet.Layers.ActivationSoftmax softmax = new Vionet.Layers.ActivationSoftmax();
        // Loss
        Vionet.LossCCE loss = new Vionet.LossCCE();

        // Optimizer
        Vionet.Optimizers.Optimizer_Adam adamOptim = new Vionet.Optimizers.Optimizer_Adam(0.9F, 0.999F, 0.001F, 0.00001F);

        // Model
        Vionet.Model model = new Vionet.Model();

        model.Add(layer1); model.Add(relu1); model.Add(dropout1);
        model.Add(layer2); model.Add(relu2); model.Add(dropout2);
        //model.Add(layer3); model.Add(relu3);
        model.Add(layer4); model.Add(softmax);

        model.set(optimizer: adamOptim, loss: loss);

        // Split
        var (X_train, X_val, y_train, y_val) = ModelSelection.TrainValSplit(X_shuffled, y_shuffled, valSplit: 0.2f);

        // Train
        model.TrainEMNIST(X_train, y_train, epochs: 15, batchSize: 128);

        // Validate
        double acc = ModelSelection.Validate(model, X_val, y_val, labels: emnistLabels, weakThreshold: 80.0);

        // Save model
        Vionet.ModelSaver.SaveJson(Path.Combine(projectRoot, "OCR", "NeuralNetworks", "model_config_EMNIST.json"), model.Layers);
    }
}




