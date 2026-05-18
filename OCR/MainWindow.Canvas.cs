using System.Drawing;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Vionet.VisionEngine;

namespace OCR
{
    public partial class MainWindow
    {

        private void Canvas_MouseDown(object sender, MouseButtonEventArgs e) => _isDrawing = true;
        private void Canvas_MouseUp(object sender, MouseButtonEventArgs e)   => _isDrawing = false;

        private void Canvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (!_isDrawing) return;

            var pos = e.GetPosition(DrawCanvas);

            var dot = new System.Windows.Shapes.Ellipse
            {
                Width  = 15,
                Height = 15,
                Fill   = System.Windows.Media.Brushes.Black
            };

            Canvas.SetLeft(dot, pos.X - dot.Width  / 2);
            Canvas.SetTop (dot, pos.Y - dot.Height / 2);

            DrawCanvas.Children.Add(dot);
        }

        // ── Очищення ────────────────────────────────────────────────────

        private void ClearCanvas_Click(object sender, RoutedEventArgs e)
        {
            DrawCanvas.Children.Clear();
            DrawResult.Text = string.Empty;
            ConfidenceList.ItemsSource = null;
        }

        private void RecognizeDraw_Click(object sender, RoutedEventArgs e)
        {
            var modelToUse = NeuralNetworkEMNIST;

            if (ActiveModelSelector.SelectedIndex > 0)
            {
                string name = (ActiveModelSelector.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "";
                if (_customModels.TryGetValue(name, out var custom))
                    modelToUse = custom;
            }

            RemoveSegmentationBoxes();

            using Bitmap bmp = CanvasToBitmap(DrawCanvas);

            var (result, confidence) = Segmentation.RecognizeTextFromBitmap(
                bmp, modelToUse, EmnistLabels, out var boxes,
                Path.Combine(_projectRoot, "OCR", "Debug-EMNIST"));

            DrawResult.Text = result;
            ConfidenceList.ItemsSource = BuildTokenList(result, confidence);

            if (ShowSegmentationCheck.IsChecked == true)
                DrawSegmentationBoxes(boxes);
        }

        private static Bitmap CanvasToBitmap(Canvas canvas)
        {
            int w = (int)canvas.ActualWidth;
            int h = (int)canvas.ActualHeight;

            var rtb = new RenderTargetBitmap(w, h, 96d, 96d, PixelFormats.Pbgra32);

            var bg = new DrawingVisual();
            using (var dc = bg.RenderOpen())
                dc.DrawRectangle(System.Windows.Media.Brushes.White, null, new Rect(0, 0, w, h));

            rtb.Render(bg);
            rtb.Render(canvas);

            using var ms = new MemoryStream();
            var enc = new BmpBitmapEncoder();
            enc.Frames.Add(BitmapFrame.Create(rtb));
            enc.Save(ms);
            return new Bitmap(ms);
        }

        private void RemoveSegmentationBoxes()
        {
            var rects = DrawCanvas.Children
                .OfType<System.Windows.Shapes.Rectangle>()
                .ToList();

            foreach (var r in rects)
                DrawCanvas.Children.Remove(r);
        }

        private void DrawSegmentationBoxes(IEnumerable<Rectangle> boxes)
        {
            foreach (var box in boxes)
            {
                var rect = new System.Windows.Shapes.Rectangle
                {
                    Width           = box.Width,
                    Height          = box.Height,
                    Stroke          = System.Windows.Media.Brushes.Red,
                    StrokeThickness = 2,
                    Fill            = System.Windows.Media.Brushes.Transparent
                };

                Canvas.SetLeft(rect, box.X);
                Canvas.SetTop (rect, box.Y);
                DrawCanvas.Children.Add(rect);
            }
        }

        private static List<OCR.models.RecognizedToken> BuildTokenList(string result, float[] confidence)
        {
            var tokens = new List<OCR.models.RecognizedToken>();
            int ci = 0;

            foreach (char ch in result)
            {
                float conf = char.IsWhiteSpace(ch) ? 1f
                    : (ci < confidence.Length ? confidence[ci++] : 0f);

                tokens.Add(new OCR.models.RecognizedToken { Symbol = ch, Confidence = conf });

            }
            return tokens;
        }
    }
}
