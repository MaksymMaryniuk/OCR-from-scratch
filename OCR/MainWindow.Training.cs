using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using Vionet;
using Vionet.Data;
using Vionet.Layers;

namespace OCR
{
    public partial class MainWindow
    {
        private void AddLayerUI_Click(object sender, RoutedEventArgs e)
        {
            var container = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                Margin = new Thickness(0, 5, 0, 5)
            };

            var typeCombo = new ComboBox { Width = 100 };
            typeCombo.Items.Add("DENSE");
            typeCombo.Items.Add("RELU");
            typeCombo.Items.Add("SOFTMAX");
            typeCombo.Items.Add("DROPOUT");
            typeCombo.SelectedIndex = 0;

            var txtIn  = new TextBox { Width = 40, Text = "128", Margin = new Thickness(5, 0, 5, 0) };
            var txtOut = new TextBox { Width = 40, Text = "64",  Margin = new Thickness(5, 0, 5, 0) };

            var btnRemove = new Button
            {
                Content         = "✕",
                Width           = 25,
                Height          = 25,
                Foreground      = Brushes.Red,
                Background      = Brushes.Transparent,
                BorderThickness = new Thickness(0),
                FontWeight      = FontWeights.Bold,
                Cursor          = Cursors.Hand,
                ToolTip         = "Видалити цей шар"
            };

            var layerInfo = new LayerUIControls
            {
                TypeBox   = typeCombo,
                InputBox  = txtIn,
                OutputBox = txtOut
            };

            btnRemove.Click += (s, ev) =>
            {
                LayersListUI.Items.Remove(container);
                _layerControls.Remove(layerInfo);
            };

            typeCombo.SelectionChanged += (s, ev) =>
            {
                bool isDense = typeCombo.SelectedItem.ToString() == "DENSE";
                txtIn.Visibility  = isDense ? Visibility.Visible : Visibility.Collapsed;
                txtOut.Visibility = isDense ? Visibility.Visible : Visibility.Collapsed;
            };

            container.Children.Add(typeCombo);
            container.Children.Add(txtIn);
            container.Children.Add(txtOut);
            container.Children.Add(btnRemove);

            LayersListUI.Items.Add(container);
            _layerControls.Add(layerInfo);
        }

        private async void StartTrain_Click(object sender, RoutedEventArgs e)
        {
            ValidateLayerDimensions();
            var customModel = BuildModelFromUI();

            ConsoleLog.Text = "";
            LogToConsole("Ініціалізація процесу навчання...");

            var (expectedOutput, maxSamples, datasetPath, datasetName) = GetDatasetParams();

            customModel.Labels = DatasetSelector.SelectedIndex switch
            {
                0 => EmnistLabels,
                1 => EnglishLabels,
                2 => UkrainianLabels,
                _ => EmnistLabels
            };

            if (!int.TryParse(EpochsInput.Text,  out int epochs))       epochs       = 5;
            if (!int.TryParse(SamplesInput.Text, out int samplesCount)) samplesCount = 10000;
            if (samplesCount > maxSamples) samplesCount = maxSamples;


            int datasetIndex = DatasetSelector.SelectedIndex;
            try
            {
                var augConfig = GetSelectedAugmentation();
                if (!ValidateModelIO(customModel, expectedOutput, datasetName)) return;

                if (customModel.Layers.Count > 0 && customModel.Layers.Last() is not ActivationSoftmax)
                {
                    LogToConsole("Додано Softmax для класифікації.");
                    customModel.Add(new ActivationSoftmax());
                }

                float learningRate      = (float)LrSlider.Value;
                var   selectedOptimizer = CreateOptimizer(learningRate);

                customModel.set(selectedOptimizer, new Vionet.LossCCE());
                LogToConsole($"Модель для {datasetName} готова. Оптимізатор: {selectedOptimizer.GetType().Name}");

                await Task.Run(() => RunTraining(customModel, datasetName, datasetPath, samplesCount, epochs, augConfig, datasetIndex));
            }
            catch (Exception ex)
            {
                LogToConsole($"Помилка: {ex.Message}");
            }
        }

        private void ValidateLayerDimensions()
        {
            for (int i = 0; i < _layerControls.Count - 1; i++)
            {
                var curr = _layerControls[i];
                var next = _layerControls[i + 1];

                if (curr.TypeBox.SelectedItem.ToString() == "DENSE" &&
                    next.TypeBox.SelectedItem.ToString() == "DENSE" &&
                    curr.OutputBox.Text != next.InputBox.Text)
                {
                    LogToConsole($"УВАГА: Невідповідність! Вихід шару {i + 1} ({curr.OutputBox.Text}) " +
                                 $"≠ Вхід шару {i + 2} ({next.InputBox.Text})");
                }
            }
        }

        private (int expectedOut, int maxSamples, string datasetPath, string datasetName) GetDatasetParams()
            => DatasetSelector.SelectedIndex switch
            {
                0 => (47, 112800,
                      Path.Combine(_projectRoot, "Vionet", "Data", "emnist-balanced-train.csv"),
                      "EMNIST"),
                1 => (EnglishLabels.Length,   50, "", "Printed EN"),
                2 => (UkrainianLabels.Length, 50, "", "Printed UA"),
                _ => throw new InvalidOperationException("Невідомий датасет")
            };

        private bool ValidateModelIO(Vionet.Model model, int expectedOutput, string datasetName)
        {
            var first = model.Layers.OfType<Layer_Dense>().FirstOrDefault();
            if (first != null && first.Weights.GetLength(0) != 784)
            {
                LogToConsole("Помилка: Перший шар повинен мати 784 входи (28×28)!");
                return false;
            }

            var last = model.Layers.OfType<Layer_Dense>().LastOrDefault();
            if (last != null && last.Weights.GetLength(1) != expectedOutput)
            {
                LogToConsole($"Помилка: Для {datasetName} останній шар повинен мати {expectedOutput} виходів!");
                return false;
            }
            return true;
        }

        private Vionet.Optimizers.Optimizer CreateOptimizer(float lr) =>
            OptChoice.SelectedIndex switch
            {
                1 => new Vionet.Optimizers.Optimizer_RMSProp(rho: 0.9f, learningRate: lr),
                2 => new Vionet.Optimizers.Optimizer_SGD(lr),
                _ => new Vionet.Optimizers.Optimizer_Adam(learningRate: lr)
            };

        private void RunTraining(
            Vionet.Model model,
            string datasetName, string datasetPath,
            int samplesCount, int epochs,
            AugmentationConfig augConfig,
            int datasetIndex)
        {
            void Report(string msg) => Dispatcher.Invoke(() => LogToConsole(msg));

            try
            {
                Report($"Завантаження {datasetName} ({samplesCount} зразків)...");
                Report("Навчання розпочато...");

                Action<int, float, float> progress = (epoch, loss, acc)
                    => Report($"🔹 Епоха {epoch}: Loss = {loss:F5}, Acc = {acc:F4}");

                switch (datasetIndex)
                {
                    case 0:
                        var (rawX, y) = Vionet.Model.LoadEMNIST(datasetPath, samplesCount);
                        float[,] X   = DataAugmentation.AugmentExistingDataset(rawX, augConfig);
                        model.Train(X, y, epochs, 64, progress);
                        break;

                    case 1:
                        var (xEn, yEn) = DatasetGenerator.GeneratePrintedData(Fonts, EnglishLabels, augConfig, samplesCount);
                        model.Train(xEn, yEn, epochs, 64, progress);
                        break;

                    case 2:
                        var (xUa, yUa) = DatasetGenerator.GeneratePrintedData(Fonts, UkrainianLabels, augConfig, samplesCount);
                        model.Train(xUa, yUa, epochs, 64, progress);
                        break;
                }

                Dispatcher.Invoke(() => FinalizeTraining(model, datasetName));
            }
            catch (Exception ex)
            {
                Report($"Помилка навчання: {ex.Message}");
            }
        }

        private void FinalizeTraining(Vionet.Model model, string datasetName)
        {
            LogToConsole("НАВЧАННЯ ЗАВЕРШЕНО!");

            string modelName = string.IsNullOrWhiteSpace(ModelNameInput.Text)
                ? $"Model_{datasetName}"
                : ModelNameInput.Text.Trim();


            Vionet.ModelSaver.SaveJson($"NeuralNetworks/{modelName}.json", model.Layers, modelName, model.Labels);

            if (!_customModels.ContainsKey(modelName))
            {
                _customModels.Add(modelName, model);
                ActiveModelSelector.Items.Add(new ComboBoxItem { Content = modelName });
                DocModelSelector.Items.Add(new ComboBoxItem { Content = modelName });
            }

            MessageBox.Show($"Модель для {datasetName} збережена!", "Готово",
                MessageBoxButton.OK, MessageBoxImage.Information);
        }

        private Vionet.Model BuildModelFromUI()
        {
            var model = new Vionet.Model();

            foreach (var ctrl in _layerControls)
            {
                switch (ctrl.TypeBox.SelectedItem.ToString())
                {
                    case "DENSE":
                        model.Add(new Layer_Dense(
                            int.Parse(ctrl.InputBox.Text),
                            int.Parse(ctrl.OutputBox.Text)));
                        break;
                    case "RELU":
                        model.Add(new ActivationReLU());
                        break;
                    case "SOFTMAX":
                        model.Add(new ActivationSoftmax());
                        break;
                    case "DROPOUT":
                        model.Add(new Layer_Dropout(0.2f));
                        break;
                }
            }

            return model;
        }

        private AugmentationConfig GetSelectedAugmentation() =>
            AugmentationSelector.SelectedIndex switch
            {
                0 => new AugmentationConfig
                {
                    UseBlur = true, BlurChance = 0.1,
                    UseSaltPepper = true, SaltPepperIntensity = 0.005,
                    UseBrightness = false, UseShift = false
                },
                1 => new AugmentationConfig
                {
                    UseBlur = true, BlurChance = 0.2,
                    UseSaltPepper = true, SaltPepperIntensity = 0.01,
                    UseBrightness = true, UseShift = true, MaxShift = 2
                },
                _ => new AugmentationConfig
                {
                    UseBlur = true, BlurChance = 0.3,
                    UseSaltPepper = true, SaltPepperIntensity = 0.02,
                    UseBrightness = true, UseShift = true, MaxShift = 4
                }
            };

        internal class LayerUIControls
        {
            public ComboBox TypeBox   { get; set; }
            public TextBox  InputBox  { get; set; }
            public TextBox  OutputBox { get; set; }
        }
    }
}
