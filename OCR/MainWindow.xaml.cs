using System.IO;
using System.Windows;
using Vionet;

namespace OCR
{
    public partial class MainWindow : Window
    {
        private bool _isDrawing = false;

        private Vionet.Model NeuralNetworkEMNIST;
        private Vionet.Model NeuralNetworkEnglish;
        private Vionet.Model NeuralNetworkUkrainian;

        private Dictionary<string, Vionet.Model> _customModels = new();
        private List<LayerUIControls> _layerControls = new();

        private List<Vionet.VisionEngine.DocumentLayoutAnalyzer.DocumentRegion> _lastRegions = null;
        private System.Drawing.Bitmap _lastOriginalBitmap = null;
        private string _loadedImagePath = "";

        private readonly string _projectRoot =
            Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\..\"));

        internal readonly string EnglishLabels  = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;()\"'";
        internal readonly string EmnistLabels   = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";
        internal readonly string UkrainianLabels = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиіїйклмнопрстуфхцчшщьюя0123456789.,!?:;()-\"'+= ";

        internal readonly string[] Fonts = { "Arial", "Times New Roman", "Verdana", "Courier New", "Calibri", "Tahoma", "Georgia" };

        public MainWindow()
        {
            InitializeComponent();

            DatasetSelector.SelectionChanged += (s, e) => UpdateDatasetHint();
            AugmentationSelector.SelectedIndex = 0;

            SyncModelsFromFolder();
            UpdateDatasetHint();

            NeuralNetworkEnglish  = Vionet.ModelSaver.LoadJson("NeuralNetworks/model_config_printed3.json");
            NeuralNetworkEMNIST   = Vionet.ModelSaver.LoadJson("NeuralNetworks/model_config_EMNIST.json");
            NeuralNetworkUkrainian = Vionet.ModelSaver.LoadJson("NeuralNetworks/model_config_printedCyrrilic.json");
        }
    }
}
