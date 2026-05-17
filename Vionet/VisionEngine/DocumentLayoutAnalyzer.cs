using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace Vionet.VisionEngine
{
    public static class DocumentLayoutAnalyzer
    {
        // =====================================================================
        // ТИПИ РЕГІОНІВ
        // =====================================================================

        public enum RegionType { Text, Image, MathFormula }

        public class DocumentRegion
        {
            public Rectangle Rect;
            public RegionType Type;
        }

        // =====================================================================
        // ПУБЛІЧНІ МЕТОДИ
        // =====================================================================

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
                    var strips = Segmentation.GetSegments(hProj, threshold: 0);

                    foreach (var strip in strips)
                    {
                        int stripH = strip.end - strip.start + 1;
                        if (stripH < 5) continue;

                        int stripPad = 10;
                        Rectangle stripRect = new Rectangle(
                            0,
                            Math.Max(0, strip.start - stripPad),
                            binary.Width,
                            Math.Min(binary.Height - strip.start, stripH + stripPad * 2));

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

        public static List<string> ExtractImages(
            Bitmap original, List<DocumentRegion> regions, string outputFolder)
        {
            System.IO.Directory.CreateDirectory(outputFolder);
            var paths = new List<string>();
            int idx = 1;

            foreach (var region in regions.Where(r => r.Type != RegionType.Text))
            {
                int pad = 12;
                Rectangle r = new Rectangle(
                    Math.Max(0, region.Rect.X - pad),
                    Math.Max(0, region.Rect.Y - pad),
                    Math.Min(original.Width - region.Rect.X, region.Rect.Width + pad * 2),
                    Math.Min(original.Height - region.Rect.Y, region.Rect.Height + pad * 2));

                string prefix = region.Type == RegionType.MathFormula ? "formula" : "image";
                string path = System.IO.Path.Combine(outputFolder, $"{prefix}{idx}.png");

                using (Bitmap crop = original.Clone(r, original.PixelFormat))
                    crop.Save(path, System.Drawing.Imaging.ImageFormat.Png);

                paths.Add(path);
                idx++;
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
                    Color baseColor = region.Type switch
                    {
                        RegionType.Text => Color.Green,
                        RegionType.Image => Color.Blue,
                        RegionType.MathFormula => Color.Orange,
                        _ => Color.Gray
                    };

                    using (SolidBrush brush = new SolidBrush(Color.FromArgb(60, baseColor)))
                        g.FillRectangle(brush, region.Rect);

                    using (Pen pen = new Pen(baseColor, 2f))
                        g.DrawRectangle(pen, region.Rect);
                }
            }

            return debugBitmap;
        }

        public static Bitmap GetMaskedTextBitmap(Bitmap original, List<DocumentRegion> regions)
        {
            Bitmap masked = new Bitmap(original);
            using (Graphics g = Graphics.FromImage(masked))
            {
                foreach (var region in regions.Where(r => r.Type != RegionType.Text))
                    g.FillRectangle(Brushes.White, region.Rect);
            }
            return masked;
        }

        // =====================================================================
        // КЛАСИФІКАЦІЯ БЛОКУ
        // =====================================================================

        static RegionType ClassifyBlock(Bitmap binary, Bitmap gray)
        {
            var components = Segmentation.GetConnectedComponents(binary);
            if (components.Count == 0) return RegionType.Text;

            if (IsMathFormula(components, binary, binary.Height))
                return RegionType.MathFormula;

            int totalPixels = binary.Width * binary.Height;
            double maxArea = components.Max(c => c.Pixels.Count);
            double avgArea = components.Average(c => c.Pixels.Count);
            double density = components.Sum(c => c.Pixels.Count) / (double)totalPixels;

            if (components.Count > 5) return RegionType.Text;

            if (maxArea > totalPixels * 0.33 && components.Count <= 3)
                return RegionType.Image;

            if (density > 0.02 && density < 0.7 && components.Count >= 2)
                return RegionType.Text;

            return RegionType.Text;
        }
        // =====================================================================
        // ДЕТЕКЦІЯ МАТЕМАТИЧНИХ ФОРМУЛ
        // =====================================================================

        static bool IsMathFormula(
    List<Segmentation.ConnectedComponent> components,
    Bitmap binary, int blockHeight)
        {
            if (components.Count < 2) return false;

            // 1. Риска дробу — широкий і тонкий компонент
            bool hasFractionBar = components.Any(c =>
            {
                float aspectRatio = (float)c.Rect.Width / Math.Max(1, c.Rect.Height);
                bool veryWide = aspectRatio > 4.0f;
                bool thin = c.Rect.Height <= 4;
                bool spansBlock = (float)c.Rect.Width / binary.Width > 0.15f;

                if (!veryWide || !thin || !spansBlock) return false;

                // ЗМІНА: Перевіряємо, чи компоненти знаходяться СУВОРО над і під рискою (перетин по осі X)
                bool hasAbove = components.Any(other =>
                    other != c &&
                    other.Rect.Bottom <= c.Rect.Top + 2 &&
                    !(other.Rect.Right < c.Rect.Left || other.Rect.Left > c.Rect.Right)); // Перевірка перетину по X

                bool hasBelow = components.Any(other =>
                    other != c &&
                    other.Rect.Top >= c.Rect.Bottom - 2 &&
                    !(other.Rect.Right < c.Rect.Left || other.Rect.Left > c.Rect.Right)); // Перевірка перетину по X

                return hasAbove && hasBelow;
            });

            // 2. Висока вертикальна дисперсія центрів
            var centerYs = components.Select(c => (double)(c.Rect.Y + c.Rect.Height / 2)).ToList();
            double meanY = centerYs.Average();
            double varianceY = centerYs.Average(y => (y - meanY) * (y - meanY));
            bool highVerticalSpread = varianceY > blockHeight * blockHeight * 0.04;

            // 3. Символи на різних вертикальних рівнях (верх/середина/низ)
            int zone = blockHeight / 3;
            bool hasTop = components.Any(c => c.Rect.Y < zone);
            bool hasBottom = components.Any(c => c.Rect.Bottom > blockHeight - zone);
            bool hasMiddle = components.Any(c => c.Rect.Y >= zone && c.Rect.Bottom <= blockHeight - zone);
            bool multiLevel = (hasTop && hasBottom) ||
                              (hasTop && hasMiddle && components.Count > 3);


            bool hasMathSymbol = false;
            var sortedByX = components.OrderBy(c => c.Rect.X).ToList();
            for (int i = 0; i < sortedByX.Count - 1 && !hasMathSymbol; i++)
            {
                var a = sortedByX[i];
                var b = sortedByX[i + 1];
                if (a.Rect.Height <= 4 && b.Rect.Height <= 4 &&
                    Math.Abs(a.Rect.Width - b.Rect.Width) < 3 &&
                    Math.Abs(a.Rect.X - b.Rect.X) < 5 &&
                    Math.Abs(a.Rect.Y - b.Rect.Y) > 2)
                    hasMathSymbol = true;
            }

            if (!hasFractionBar && !hasMathSymbol)
                return false;

            int score = (hasFractionBar ? 3 : 0) +
                        (highVerticalSpread ? 2 : 0) +
                        (multiLevel ? 2 : 0) +
                        (hasMathSymbol ? 1 : 0);

            return score >= 4;
        }

        // =====================================================================
        // ПРОЕКЦІЇ І ЗЛИТТЯ РЕГІОНІВ
        // =====================================================================

        static int[] GetVerticalProjection(Bitmap bmp, Rectangle rect)
        {
            int[] projection = new int[rect.Width];
            for (int x = 0; x < rect.Width; x++)
                for (int y = 0; y < rect.Height; y++)
                    if (bmp.GetPixel(rect.X + x, rect.Y + y).R < 128)
                        projection[x]++;
            return projection;
        }

        static List<(int start, int end)> GetVerticalBlocks(int[] projection, int minGap)
        {
            var blocks = new List<(int, int)>();
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
                        if (regions[i].Type != regions[j].Type) continue;

                        Rectangle r1 = regions[i].Rect;
                        r1.Inflate(15, 15);

                        if (r1.IntersectsWith(regions[j].Rect))
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
                    if (changed) break;
                }
            }
            return regions;
        }
    }
}