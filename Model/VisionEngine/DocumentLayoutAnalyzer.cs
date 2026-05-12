using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace Model.VisionEngine
{
    public static class DocumentLayoutAnalyzer
    {
        public enum RegionType { Text, Image }

        public class DocumentRegion
        {
            public Rectangle Rect;
            public RegionType Type;
        }

        public static List<DocumentRegion> AnalyzeLayout(Bitmap original)
        {
            var regions = new List<DocumentRegion>();

            using (Bitmap scaled = ImagePreprocessing.ScaleImageForOCR(original))
            using (Bitmap gray = ImagePreprocessing.ToGrayscale(scaled))
            {
                int otsu = ImagePreprocessing.GetOtsuThreshold(gray);
                using (Bitmap binary = ImagePreprocessing.Threshold(gray, otsu))
                {
                    var hProj = Segmentation.GetHorizontalProjection(binary);
                    var strips = Segmentation.GetSegments(hProj, threshold: 1);

                    foreach (var strip in strips)
                    {
                        int stripH = strip.end - strip.start + 1;
                        if (stripH < 5) continue;

                        Rectangle stripRect = new Rectangle(0, strip.start, binary.Width, stripH);

                        int[] vProj = GetVerticalProjection(binary, stripRect);
                        
                        var blocks = GetVerticalBlocks(vProj, minGap: 15);

                        foreach (var block in blocks)
                        {
                            int blockW = block.end - block.start + 1;
                            if (blockW < 5) continue;

                            Rectangle blockRect = new Rectangle(block.start, strip.start, blockW, stripH);

                            using (Bitmap binBlock = binary.Clone(blockRect, binary.PixelFormat))
                            using (Bitmap grayBlock = gray.Clone(blockRect, gray.PixelFormat))
                            {
                                var type = ClassifyBlock(binBlock, grayBlock);
                                regions.Add(new DocumentRegion { Rect = blockRect, Type = type });
                            }
                        }
                    }

                    regions = MergeAdjacentRegions(regions);
                }
            }

            return regions;
        }

        static RegionType ClassifyBlock(Bitmap binary, Bitmap gray)
        {
            var components = Segmentation.GetConnectedComponents(binary);
            if (components.Count == 0) return RegionType.Text;

            int totalPixels = binary.Width * binary.Height;
            double avgArea = components.Average(c => c.Pixels.Count);
            double maxArea = components.Max(c => c.Pixels.Count);
            double variance = GetVariance(gray);

            // Фікс 1: hugeBlob тепер 25% замість 12% — підкреслення не тригерить
            bool hugeBlob = maxArea > totalPixels * 0.25;
            bool highVariance = variance > 3000;
            bool tallStrip = binary.Height > 80;

            // Фікс 2: manySmall враховує розмір блоку — для вузьких блоків достатньо 2 компоненти
            int minComponents = binary.Width > 100 ? 4 : 2;
            bool manySmall = components.Count >= minComponents && avgArea < 600;
            bool lowVariance = variance < 1800;

            // Фікс 3: щільність тексту — символи займають 5-60% площі
            double density = components.Sum(c => c.Pixels.Count) / (double)totalPixels;
            bool textDensity = density > 0.03 && density < 0.6;

            int textScore = (manySmall ? 2 : 0) +
                             (lowVariance ? 1 : 0) +
                             (textDensity ? 1 : 0);

            int imageScore = (hugeBlob ? 2 : 0) +
                             (highVariance ? 2 : 0) +
                             (tallStrip ? 1 : 0);

            // Фікс 4: якщо компонентів мало але є textDensity → текст (одне слово)
            if (components.Count <= 6 && textDensity && !hugeBlob)
                return RegionType.Text;

            return textScore >= imageScore ? RegionType.Text : RegionType.Image;
        }

        static int[] GetVerticalProjection(Bitmap bmp, Rectangle rect)
        {
            int[] projection = new int[rect.Width];
            for (int x = 0; x < rect.Width; x++)
            {
                for (int y = 0; y < rect.Height; y++)
                {
                    if (bmp.GetPixel(rect.X + x, rect.Y + y).R < 128)
                        projection[x]++;
                }
            }
            return projection;
        }

        static List<(int start, int end)> GetVerticalBlocks(int[] projection, int minGap)
        {
            List<(int, int)> blocks = new();
            int start = -1;
            int gapCount = 0;

            for (int i = 0; i < projection.Length; i++)
            {
                if (projection[i] > 0)
                {
                    if (start == -1) start = i;
                    gapCount = 0;
                }
                else if (start != -1)
                {
                    gapCount++;
                    if (gapCount >= minGap)
                    {
                        blocks.Add((start, i - gapCount));
                        start = -1;
                    }
                }
            }
            if (start != -1) blocks.Add((start, projection.Length - 1));
            return blocks;
        }


        static List<DocumentRegion> MergeAdjacentRegions(List<DocumentRegion> regions)
        {
            if (regions.Count == 0) return regions;

            bool changed = true;
            while (changed)
            {
                changed = false;
                for (int i = 0; i < regions.Count; i++)
                {
                    for (int j = i + 1; j < regions.Count; j++)
                    {
                        if (regions[i].Type == regions[j].Type)
                        {
                            Rectangle r1 = regions[i].Rect;
                            Rectangle r2 = regions[j].Rect;

                            r1.Inflate(15, 15);
                            if (r1.IntersectsWith(r2))
                            {
                                int minX = Math.Min(regions[i].Rect.X, regions[j].Rect.X);
                                int minY = Math.Min(regions[i].Rect.Y, regions[j].Rect.Y);
                                int maxX = Math.Max(regions[i].Rect.Right, regions[j].Rect.Right);
                                int maxY = Math.Max(regions[i].Rect.Bottom, regions[j].Rect.Bottom);

                                regions[i].Rect = new Rectangle(minX, minY, maxX - minX, maxY - minY);
                                regions.RemoveAt(j);
                                changed = true;
                                break;
                            }
                        }
                    }
                    if (changed) break;
                }
            }
            return regions;
        }

        static double GetVariance(Bitmap bmp)
        {
            double sum = 0, sumSq = 0;
            int count = bmp.Width * bmp.Height;
            for (int y = 0; y < bmp.Height; y++)
                for (int x = 0; x < bmp.Width; x++)
                {
                    double v = bmp.GetPixel(x, y).R;
                    sum += v; sumSq += v * v;
                }
            double mean = sum / count;
            return (sumSq / count) - mean * mean;
        }

        public static List<string> ExtractImages(Bitmap original, List<DocumentRegion> regions, string outputFolder)
        {
            System.IO.Directory.CreateDirectory(outputFolder);
            var paths = new List<string>();
            int idx = 1;

            foreach (var region in regions.Where(r => r.Type == RegionType.Image))
            {
                int pad = 4;
                Rectangle r = new Rectangle(
                    Math.Max(0, region.Rect.X - pad),
                    Math.Max(0, region.Rect.Y - pad),
                    Math.Min(original.Width - region.Rect.X, region.Rect.Width + pad * 2),
                    Math.Min(original.Height - region.Rect.Y, region.Rect.Height + pad * 2));

                using (Bitmap crop = original.Clone(r, original.PixelFormat))
                {
                    string path = System.IO.Path.Combine(outputFolder, $"image{idx}.png");
                    crop.Save(path, System.Drawing.Imaging.ImageFormat.Png);
                    paths.Add(path);
                    idx++;
                }
            }
            return paths;
        }

        public static Bitmap GetLayoutDebugBitmap(Bitmap original, List<DocumentRegion> regions)
        {
            Bitmap debugBitmap = new Bitmap(original);

            using (Graphics g = Graphics.FromImage(debugBitmap))
            using (Font font = new Font("Arial", 12f, System.Drawing.FontStyle.Bold))
            {
                foreach (var region in regions)
                {
                    bool isImage = region.Type == RegionType.Image;

                    Color baseColor = isImage ? Color.Blue : Color.Green;

                    Color fillColor = Color.FromArgb(60, baseColor);

                    using (SolidBrush brush = new SolidBrush(fillColor))
                    {
                        g.FillRectangle(brush, region.Rect);
                    }

                    using (Pen pen = new Pen(baseColor, 2f))
                    {
                        g.DrawRectangle(pen, region.Rect);
                    }

                }
            }

            return debugBitmap;
        }


        public static Bitmap GetMaskedTextBitmap(Bitmap original, List<DocumentRegion> regions)
        {
            // Створюємо копію зображення, щоб не зіпсувати оригінал
            Bitmap masked = new Bitmap(original);

            using (Graphics g = Graphics.FromImage(masked))
            {
                // Проходимось по всіх регіонах, які класифіковано як Image
                foreach (var region in regions.Where(r => r.Type == RegionType.Image))
                {
                    // Замальовуємо їх білим кольором
                    g.FillRectangle(Brushes.White, region.Rect);
                }
            }
            return masked;
        }
    }
}