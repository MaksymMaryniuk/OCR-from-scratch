using Microsoft.Win32;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Model;
using System.Drawing;
using System.IO;

namespace OCR
{
    public partial class MainWindow : Window
    {

        WriteableBitmap drawingBitmap;
        bool isDrawing = false;

        private Model.Model NeuralNetworkEMNIST;
        private Model.Model NeuralNetworkPrinted;
        private string loadedImagePath = "";

        private string printedLabels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        private string emnistLabels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";
        public MainWindow()
        {
            InitializeComponent();
            InitCanvas();
            InitializeModel(ref NeuralNetworkPrinted, "NeuralNetworks/model_config_printed.json");
            InitializeModel(ref NeuralNetworkEMNIST, "NeuralNetworks/model_config_EMNIST.json");
        }
        private void InitCanvas()
        {
            int width = 600;
            int height = 280;

            drawingBitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
            DrawImage.Source = drawingBitmap;
        }


        private void DrawImage_MouseDown(object sender, MouseButtonEventArgs e)
        {
            isDrawing = true;
        }

        private void DrawImage_MouseMove(object sender, MouseEventArgs e)
        {
            if (!isDrawing || e.LeftButton != MouseButtonState.Pressed) return;

            var pos = e.GetPosition(DrawImage);
            DrawCircle((int)pos.X, (int)pos.Y, 10);
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
                Width = 10,
                Height = 10,
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

        private void DrawCircle(int cx, int cy, int radius)
        {
            drawingBitmap.Lock();

            unsafe
            {
                IntPtr pBackBuffer = drawingBitmap.BackBuffer;
                int stride = drawingBitmap.BackBufferStride;

                for (int y = -radius; y <= radius; y++)
                {
                    for (int x = -radius; x <= radius; x++)
                    {
                        if (x * x + y * y <= radius * radius)
                        {
                            int px = cx + x;
                            int py = cy + y;

                            if (px >= 0 && px < drawingBitmap.PixelWidth &&
                                py >= 0 && py < drawingBitmap.PixelHeight)
                            {
                                byte* pixel = (byte*)pBackBuffer + py * stride + px * 4;

                                pixel[0] = 0;   // B
                                pixel[1] = 0;   // G
                                pixel[2] = 0;   // R
                                pixel[3] = 255; // A
                            }
                        }
                    }
                }
            }

            drawingBitmap.AddDirtyRect(new Int32Rect(0, 0, drawingBitmap.PixelWidth, drawingBitmap.PixelHeight));
            drawingBitmap.Unlock();
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
            if (NeuralNetworkPrinted == null) return;

            //string result = Segmentation.RecognizeMultiLineText(userDrawing, nn, printedLabels);
            string resultimage = Segmentation.RecognizeTextFromPhoto(loadedImagePath, NeuralNetworkPrinted, printedLabels, "DEBUG-PRINTED");


            ResultText.Text = $"Результат: {resultimage}";
        }


        private void RecognizeDraw_Click(object sender, RoutedEventArgs e)
        {
            var oldRectangles = DrawCanvas.Children.OfType<System.Windows.Shapes.Rectangle>().ToList();
            foreach (var rect in oldRectangles)
            {
                DrawCanvas.Children.Remove(rect);
            }

            using (Bitmap bmp = CanvasToBitmap(DrawCanvas))
            {
                List<System.Drawing.Rectangle> boxes;
                string result = Segmentation.RecognizeTextFromBitmap(bmp, NeuralNetworkEMNIST, emnistLabels, out boxes, "DEBUG-EMNIST");
                DrawResult.Text = result;

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

            rtb.Render(canvas);

            MemoryStream ms = new MemoryStream();
            BitmapEncoder encoder = new BmpBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(rtb));
            encoder.Save(ms);


            Bitmap bitmap = new Bitmap(ms);

            return bitmap;
        }


        private void InitializeModel(ref Model.Model nn, string path)
        {
            var loadedDenseLayers = Model.ModelSaver.LoadJson(path);

            nn = new Model.Model();

            nn.Add(loadedDenseLayers[0]);
            nn.Add(new Model.Layers.ActivationReLU());

            nn.Add(loadedDenseLayers[1]);
            nn.Add(new Model.Layers.ActivationReLU());

            nn.Add(loadedDenseLayers[2]);
            nn.Add(new Model.Layers.ActivationSoftmax());
        }

    }
}