using Microsoft.Win32;
using System.Drawing;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Vionet.VisionEngine;
using Brushes = System.Windows.Media.Brushes;

namespace OCR
{
    public partial class MainWindow : Window
    {
        bool isDrawing = false;

        private Vionet.Model NeuralNetworkEMNIST;
        private Vionet.Model NeuralNetworkPrinted;
        private Vionet.Model NeuralNetworkUkrainian;

        private List<DocumentLayoutAnalyzer.DocumentRegion> _lastRegions = null;
        private Bitmap _lastOriginalBitmap = null;

        private string loadedImagePath = "";
        private string projectRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\..\"));

        private string printedLabels = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;()\"'";
        private string emnistLabels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";
        private string ukrainianLabels = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиіїйклмнопрстуфхцчшщьюя0123456789.,!?:;()-\"'+= ";

        public MainWindow()
        {
            InitializeComponent();
            InitCanvas();
            NeuralNetworkPrinted = Vionet.ModelSaver.LoadJson("NeuralNetworks/model_config_printed3.json");
            NeuralNetworkEMNIST = Vionet.ModelSaver.LoadJson("NeuralNetworks/model_config_EMNIST.json");
            NeuralNetworkUkrainian = Vionet.ModelSaver.LoadJson("NeuralNetworks/model_config_printedCyrrilic.json");

        }
        private void InitCanvas()
        {
            int width = 600;
            int height = 280;
        }

        private void Canvas_MouseUp(object sender, MouseButtonEventArgs e)
        {
            isDrawing = false;
        }

        private void Canvas_MouseDown(object sender, MouseButtonEventArgs e)
        {
            isDrawing = true;
        }


        private void Canvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (!isDrawing) return;

            var pos = e.GetPosition(DrawCanvas);

            var ellipse = new System.Windows.Shapes.Ellipse
            {
                Width = 15,
                Height = 15,
                Fill = System.Windows.Media.Brushes.Black
            };

            Canvas.SetLeft(ellipse, pos.X);
            Canvas.SetTop(ellipse, pos.Y);

            DrawCanvas.Children.Add(ellipse);
        }

        private void ClearCanvas_Click(object sender, RoutedEventArgs e)
        {
            DrawCanvas.Children.Clear();

            DrawResult.Text = "";
        }
        private void ImageDropZone_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                e.Effects = DragDropEffects.Copy;

                ImageDropZone.Background =
                    new SolidColorBrush(System.Windows.Media.Color.FromArgb(40, 0, 120, 255));
            }
        }
        private void ImageDropZone_Drop(object sender, DragEventArgs e)
        {
            ImageDropZone.Background = Brushes.Transparent;

            if (!e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                MessageBox.Show(
                    "Файл не вдалося отримати.",
                    "Помилка",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);

                return;
            }

            string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);

            if (files.Length == 0)
                return;

            string file = files[0];

            // Перевірка розширення
            string ext = Path.GetExtension(file).ToLower();

            bool isImage =
                ext == ".png" ||
                ext == ".jpg" ||
                ext == ".jpeg" ||
                ext == ".bmp";

            if (!isImage)
            {
                MessageBox.Show(
                    "Потрібно вибрати файл зображення (.png, .jpg, .bmp).",
                    "Непідтримуваний формат",
                    MessageBoxButton.OK,
                    MessageBoxImage.Warning);

                return;
            }

            try
            {
                loadedImagePath = file;

                BitmapImage bitmap = new BitmapImage();

                bitmap.BeginInit();
                bitmap.UriSource = new Uri(file);
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.EndInit();

                LoadedImage.Source = bitmap;

                DropHint.Visibility = Visibility.Collapsed;
            }
            catch
            {
                MessageBox.Show(
                    "Не вдалося відкрити зображення.",
                    "Помилка завантаження",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);
            }
        }

        private void ImageDropZone_DragLeave(object sender, DragEventArgs e)
        {
            ImageDropZone.Background = Brushes.Transparent;
        }

        private void LoadImage_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dlg = new OpenFileDialog();
            dlg.Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp";

            if (dlg.ShowDialog() == true)
            {
                loadedImagePath = dlg.FileName;
                LoadedImage.Source = new BitmapImage(new Uri(loadedImagePath));
            }
        }

        void Show(Grid grid)
        {
            MainMenu.Visibility = Visibility.Collapsed;
            RunView.Visibility = Visibility.Collapsed;
            TestView.Visibility = Visibility.Collapsed;

            grid.Visibility = Visibility.Visible;
        }

        private void Run_Click(object sender, RoutedEventArgs e)
        {
            Show(RunView);
        }

        private void Test_Click(object sender, RoutedEventArgs e)
        {
            Show(TestView);
        }

        private void Back_Click(object sender, RoutedEventArgs e)
        {
            Show(MainMenu);
        }

        private void Recognize_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(loadedImagePath)) return;

            var selectedModel = LanguageSelector.SelectedIndex == 0 ? NeuralNetworkPrinted : NeuralNetworkUkrainian;
            var selectedLabels = LanguageSelector.SelectedIndex == 0 ? printedLabels : ukrainianLabels;

            _lastOriginalBitmap?.Dispose();
            _lastOriginalBitmap = new Bitmap(loadedImagePath);

            _lastRegions = DocumentLayoutAnalyzer.AnalyzeLayout(_lastOriginalBitmap);

            Bitmap textOnlyBitmap = DocumentLayoutAnalyzer.GetMaskedTextBitmap(_lastOriginalBitmap, _lastRegions);

            var (text, _) = Segmentation.RecognizeTextFromBitmap(
                textOnlyBitmap, selectedModel, selectedLabels, out _,
                Path.Combine(projectRoot, "OCR", "Debug-Printed"));

            ResultText.Text = text;

            using (Bitmap debug = DocumentLayoutAnalyzer.GetLayoutDebugBitmap(_lastOriginalBitmap, _lastRegions))
                LoadedImage.Source = BitmapToBitmapSource(debug);
        }


        private void RecognizeDraw_Click(object sender, RoutedEventArgs e)
        {
            List<RecognizedToken> tokens = new List<RecognizedToken>();
            var oldRectangles = DrawCanvas.Children.OfType<System.Windows.Shapes.Rectangle>().ToList();
            foreach (var rect in oldRectangles)
            {
                DrawCanvas.Children.Remove(rect);
            }


            using (Bitmap bmp = CanvasToBitmap(DrawCanvas))
            {
                List<Rectangle> boxes;
                (string result, float[] confidence) = Segmentation.RecognizeTextFromBitmap(bmp, NeuralNetworkEMNIST, emnistLabels, out boxes,
                    Path.Combine(projectRoot, "OCR", "Debug-EMNIST"));

                DrawResult.Text = result;
                int confidenceIdx = 0;

                for (int i = 0; i < result.Length; i++)
                {
                    char currentSymbol = result[i];
                    float currentConf = 0f;

                    if (!char.IsWhiteSpace(currentSymbol))
                    {
                        if (confidenceIdx < confidence.Length)
                        {
                            currentConf = confidence[confidenceIdx];
                            confidenceIdx++;
                        }
                    }
                    else
                    {
                        currentConf = 1.0f;
                    }

                    tokens.Add(new RecognizedToken
                    {
                        Symbol = currentSymbol,
                        Confidence = currentConf
                    });
                }

                ConfidenceList.ItemsSource = tokens;

                if (ShowSegmentationCheck.IsChecked == true)
                {
                    foreach (var box in boxes)
                    {
                        var rectShape = new System.Windows.Shapes.Rectangle
                        {
                            Width = box.Width,
                            Height = box.Height,
                            Stroke = System.Windows.Media.Brushes.Red,
                            StrokeThickness = 2,
                            Fill = System.Windows.Media.Brushes.Transparent
                        };

                        Canvas.SetLeft(rectShape, box.X);
                        Canvas.SetTop(rectShape, box.Y);

                        DrawCanvas.Children.Add(rectShape);
                    }
                }
            }
        }


        private Bitmap CanvasToBitmap(Canvas canvas)
        {
            int width = (int)canvas.ActualWidth;
            int height = (int)canvas.ActualHeight;

            RenderTargetBitmap rtb = new RenderTargetBitmap(
                width, height, 96d, 96d, PixelFormats.Pbgra32);

            DrawingVisual background = new DrawingVisual();
            using (DrawingContext dc = background.RenderOpen())
            {
                dc.DrawRectangle(
                    System.Windows.Media.Brushes.White, null,
                    new Rect(0, 0, width, height));
            }
            rtb.Render(background);
            rtb.Render(canvas);

            MemoryStream ms = new MemoryStream();
            BitmapEncoder encoder = new BmpBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(rtb));
            encoder.Save(ms);

            return new Bitmap(ms);
        }



        public class RecognizedToken
        {
            public char Symbol { get; set; }
            public float Confidence { get; set; }
            public SolidColorBrush ConfidenceBrush
            {
                get
                {
                    byte red = (byte)(255 * (1 - Confidence));
                    byte green = (byte)(255 * Confidence);
                    return new SolidColorBrush(System.Windows.Media.Color.FromRgb(red, green, 0));
                }
            }
        }

        private BitmapSource BitmapToBitmapSource(Bitmap bmp)
        {
            using (MemoryStream ms = new MemoryStream())
            {
                bmp.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
                ms.Position = 0;

                BitmapImage bitmapImage = new BitmapImage();
                bitmapImage.BeginInit();
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.StreamSource = ms;
                bitmapImage.EndInit();
                bitmapImage.Freeze();
                return bitmapImage;
            }
        }


        private void SaveToFile_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(ResultText.Text))
            {
                MessageBox.Show("Немає тексту для збереження!", "Увага", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "Текстові файли (*.txt)|*.txt";
            sfd.FileName = "Результат_OCR";

            if (sfd.ShowDialog() == true)
            {
                File.WriteAllText(sfd.FileName, ResultText.Text);
                MessageBox.Show("Текст збережено.");
            }
        }

        private void SaveImages_Click(object sender, RoutedEventArgs e)
        {
            if (_lastRegions == null || _lastOriginalBitmap == null)
            {
                MessageBox.Show("Спочатку виконайте розпізнавання!", "Увага",
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var nonTextRegions = _lastRegions
                .Where(r => r.Type != DocumentLayoutAnalyzer.RegionType.Text)
                .ToList();

            if (nonTextRegions.Count == 0)
            {
                MessageBox.Show("На зображенні не знайдено фото або формул.", "Увага",
                    MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var dialog = new Microsoft.Win32.OpenFolderDialog
            {
                Title = "Оберіть папку для збереження знайдених зображень"
            };

            if (dialog.ShowDialog() != true) return;

            var saved = DocumentLayoutAnalyzer.ExtractImages(
                _lastOriginalBitmap, _lastRegions, dialog.FolderName);

            MessageBox.Show(
                $"Збережено {saved.Count} файл(ів):\n" +
                string.Join("\n", saved.Select(Path.GetFileName)),
                "Готово", MessageBoxButton.OK, MessageBoxImage.Information);
        }
    }
}