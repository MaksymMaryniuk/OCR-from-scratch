using Microsoft.Win32;
using System.Drawing;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Vionet;
using Vionet.VisionEngine;

namespace OCR
{
    public partial class MainWindow
    {
        private void ImageDropZone_DragEnter(object sender, DragEventArgs e)
        {
            if (!e.Data.GetDataPresent(DataFormats.FileDrop)) return;

            e.Effects = DragDropEffects.Copy;
            ImageDropZone.Background =
                new SolidColorBrush(System.Windows.Media.Color.FromArgb(40, 0, 120, 255));
        }

        private void ImageDropZone_DragLeave(object sender, DragEventArgs e)
            => ImageDropZone.Background = System.Windows.Media.Brushes.Transparent;

        private void ImageDropZone_Drop(object sender, DragEventArgs e)
        {
            ImageDropZone.Background = System.Windows.Media.Brushes.Transparent;

            if (!e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                ShowError("Файл не вдалося отримати.", "Помилка");
                return;
            }

            var files = (string[])e.Data.GetData(DataFormats.FileDrop);
            if (files.Length == 0) return;

            TryLoadImage(files[0]);
        }

        private void LoadImage_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog { Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp" };
            if (dlg.ShowDialog() == true)
                TryLoadImage(dlg.FileName);
        }

        private void Recognize_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_loadedImagePath)) return;

            var (model, labels) = SelectDocumentModel();

            _lastOriginalBitmap?.Dispose();
            _lastOriginalBitmap = new Bitmap(_loadedImagePath);

            _lastRegions = DocumentLayoutAnalyzer.AnalyzeLayout(_lastOriginalBitmap);

            using var textBitmap = DocumentLayoutAnalyzer.GetMaskedTextBitmap(_lastOriginalBitmap, _lastRegions);

            var (text, _) = Segmentation.RecognizeTextFromBitmap(
                textBitmap, model, labels, out _,
                Path.Combine(_projectRoot, "OCR", "Debug-Printed"));

            ResultText.Text = text;

            using var debug = DocumentLayoutAnalyzer.GetLayoutDebugBitmap(_lastOriginalBitmap, _lastRegions);
            LoadedImage.Source = BitmapToBitmapSource(debug);
        }

        private void SaveToFile_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(ResultText.Text))
            {
                MessageBox.Show("Немає тексту для збереження!", "Увага",
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var sfd = new SaveFileDialog
            {
                Filter   = "Текстові файли (*.txt)|*.txt",
                FileName = "Результат_OCR"
            };

            if (sfd.ShowDialog() == true)
            {
                File.WriteAllText(sfd.FileName, ResultText.Text);
                MessageBox.Show("Текст збережено.", "Готово",
                    MessageBoxButton.OK, MessageBoxImage.Information);
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

            var nonText = _lastRegions
                .Where(r => r.Type != DocumentLayoutAnalyzer.RegionType.Text)
                .ToList();

            if (nonText.Count == 0)
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

        private void TryLoadImage(string path)
        {
            string ext = Path.GetExtension(path).ToLower();
            bool isImage = ext is ".png" or ".jpg" or ".jpeg" or ".bmp";

            if (!isImage)
            {
                MessageBox.Show("Потрібно вибрати файл зображення (.png, .jpg, .bmp).",
                    "Непідтримуваний формат", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            try
            {
                _loadedImagePath = path;

                var bmp = new BitmapImage();
                bmp.BeginInit();
                bmp.UriSource    = new Uri(path);
                bmp.CacheOption  = BitmapCacheOption.OnLoad;
                bmp.EndInit();

                LoadedImage.Source    = bmp;
                DropHint.Visibility   = Visibility.Collapsed;
            }
            catch
            {
                ShowError("Не вдалося відкрити зображення.", "Помилка завантаження");
            }
        }

        private (Vionet.Model model, string labels) SelectDocumentModel()
        {
            if (DocModelSelector.SelectedIndex > 1)
            {
                string name = (DocModelSelector.SelectedItem as ComboBoxItem)!.Content.ToString()!;
                return (_customModels[name], EmnistLabels);
            }

            return DocModelSelector.SelectedIndex == 1
                ? (NeuralNetworkEnglish,   EnglishLabels)
                : (NeuralNetworkUkrainian, UkrainianLabels);
        }

        internal static BitmapSource BitmapToBitmapSource(Bitmap bmp)
        {
            using var ms = new MemoryStream();
            bmp.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
            ms.Position = 0;

            var img = new BitmapImage();
            img.BeginInit();
            img.CacheOption  = BitmapCacheOption.OnLoad;
            img.StreamSource = ms;
            img.EndInit();
            img.Freeze();
            return img;
        }

        private static void ShowError(string message, string title)
            => MessageBox.Show(message, title, MessageBoxButton.OK, MessageBoxImage.Error);
    }
}
