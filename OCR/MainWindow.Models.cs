using System.IO;
using Vionet;

namespace OCR
{
    public partial class MainWindow
    {
        private void SyncModelsFromFolder()
        {
            string folder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "NeuralNetworks");

            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
                return;
            }

            foreach (string filePath in Directory.GetFiles(folder, "*.json"))
            {
                try
                {
                    string name = Path.GetFileNameWithoutExtension(filePath);

                    if (name.StartsWith("model_config")) continue;

                    if (_customModels.ContainsKey(name)) continue;

                    var model = Vionet.ModelSaver.LoadJson(filePath);
                    _customModels.Add(name, model);

                    ActiveModelSelector.Items.Add(new System.Windows.Controls.ComboBoxItem { Content = name });
                    DocModelSelector.Items.Add(new System.Windows.Controls.ComboBoxItem { Content = name });
                }
                catch (Exception ex)
                {
                    LogToConsole($"Помилка завантаження {filePath}: {ex.Message}");
                }
            }
        }


        private void UpdateDatasetHint()
        {
            if (DatasetSelector == null || ConsoleLog == null) return;

            var (expectedOut, maxSamples, langName) = DatasetSelector.SelectedIndex switch
            {
                0 => (47,                    112800, "EMNIST (Рукописні)"),
                1 => (EnglishLabels.Length,     100, "Друковані (EN)"),
                2 => (UkrainianLabels.Length,   100, "Друковані (UA)"),
                _ => (0, 0, "")
            };

            const int expectedIn = 784;

            ConsoleLog.Text = "";
            LogToConsole($"---  ПІДКАЗКА ДЛЯ {langName.ToUpper()} ---");
            LogToConsole($"🔹 Вхідний шар : ПЕРШИЙ DENSE повинен мати {expectedIn} входів.");
            LogToConsole($"🔹 Вихідний шар: ОСТАННІЙ DENSE повинен мати {expectedOut} виходів.");
            LogToConsole($"🔹 Макс. зразків: {maxSamples}");

            if (maxSamples == 100)
            {
                LogToConsole("🔹 Семпл множиться на кількість символів та шрифтів.");
                LogToConsole("   (Базово: 7 шрифтів × кількість символів × задана частота)");
            }

            LogToConsole("---  ПРАВИЛА ПОБУДОВИ АРХІТЕКТУРИ ---");
            LogToConsole("1. Out попереднього DENSE = In наступного DENSE.");
            LogToConsole("   Приклад: [784→128] → [128→64].");
            LogToConsole("2. Після кожного DENSE (крім останнього) — RELU.");
            LogToConsole("3. Softmax додається автоматично.");
            LogToConsole("------------------------------------------");
        }

        private void LogToConsole(string message)
        {
            ConsoleLog.AppendText($"[{DateTime.Now:HH:mm:ss}] {message}{Environment.NewLine}");
            ConsoleLog.ScrollToEnd();
        }
    }
}
