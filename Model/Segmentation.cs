using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;

namespace Model
{
    public static class Segmentation
    {
        public static string RecognizeTextFromPhoto(string imagePath, Model nn, string labels, string debugDir = null)
        {
            using (Bitmap original = new Bitmap(imagePath))
            {
                return RecognizeTextFromBitmap(original, nn, labels, out _, debugDir);
            }
        }

        public static string RecognizeTextFromBitmap(Bitmap original, Model nn, string labels, out List<Rectangle> boundingBoxes, string debugDir = null)
        {
            StringBuilder fullText = new StringBuilder();
            boundingBoxes = new List<Rectangle>();

            if (debugDir != null)
            {
                if (Directory.Exists(debugDir)) Directory.Delete(debugDir, recursive: true);
                Directory.CreateDirectory(debugDir);
            }

            int debugIndex = 0;

            using (Bitmap scaled = ImagePreprocessing.ScaleImageForOCR(original))
            using (Bitmap gray = ImagePreprocessing.ToGrayscale(scaled))
            {
                float scaleX = (float)original.Width / scaled.Width;
                float scaleY = (float)original.Height / scaled.Height;

                int optimalThreshold = ImagePreprocessing.GetOtsuThreshold(gray);
                using (Bitmap binary = ImagePreprocessing.Threshold(gray, optimalThreshold))
                {
                    var hProjection = GetHorizontalProjection(binary);
                    var lineSegments = GetSegments(hProjection, threshold: 2);

                    int descenderPad = 8;
                    int ascenderPad = 3;

                    foreach (var line in lineSegments)
                    {
                        int lineHeight = line.end - line.start + 1;
                        if (lineHeight < 10) continue;

                        int paddedStart = Math.Max(0, line.start - ascenderPad);
                        int paddedEnd = Math.Min(binary.Height - 1, line.end + descenderPad);
                        int paddedHeight = paddedEnd - paddedStart + 1;

                        Rectangle lineRect = new Rectangle(0, paddedStart, binary.Width, paddedHeight);

                        using (Bitmap lineBinary = binary.Clone(lineRect, binary.PixelFormat))
                        using (Bitmap lineGray = gray.Clone(lineRect, gray.PixelFormat))
                        {
                            var components = GetConnectedComponents(lineBinary)
                                .Where(c => c.Rect.Width >= 1 && c.Rect.Height >= 1)
                                .OrderBy(c => c.Rect.X)
                                .ToList();

                            components = MergeDotComponents(components, lineHeight);
                            components = components.OrderBy(c => c.Rect.X).ToList();

                            for (int i = 0; i < components.Count; i++)
                            {
                                var comp = components[i];

                                Rectangle absRect = new Rectangle(comp.Rect.X + lineRect.X, comp.Rect.Y + lineRect.Y, comp.Rect.Width, comp.Rect.Height);
                                Rectangle origRect = new Rectangle(
                                    (int)(absRect.X * scaleX), (int)(absRect.Y * scaleY),
                                    (int)(absRect.Width * scaleX), (int)(absRect.Height * scaleY)
                                );
                                boundingBoxes.Add(origRect);

                                using (Bitmap charBmp = lineGray.Clone(comp.Rect, lineGray.PixelFormat))
                                {
                                    var input = ImagePreprocessing.GetInputForModel(charBmp);
                                    var output = nn.Forward(input);
                                    int idx = AdditionalMath.GetArgmax(output);
                                    char predicted = labels[idx];
                                    fullText.Append(predicted);

                                    if (debugDir != null)
                                    {
                                        SaveDebugImages(charBmp, debugDir, debugIndex, predicted);
                                        debugIndex++;
                                    }
                                }

                                if (i < components.Count - 1)
                                {
                                    int gap = components[i + 1].Rect.X - comp.Rect.Right;
                                    if (gap > lineHeight * 0.4) fullText.Append(" ");
                                }
                            }
                            fullText.AppendLine();
                        }
                    }
                }
            }
            return fullText.ToString().TrimEnd();
        }


        public static List<ConnectedComponent> GetConnectedComponents(Bitmap bmp)
        {
            int w = bmp.Width; int h = bmp.Height;
            bool[,] visited = new bool[w, h];
            List<ConnectedComponent> components = new List<ConnectedComponent>();

            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    if (IsBlack(bmp.GetPixel(x, y)) && !visited[x, y])
                    {
                        var component = new ConnectedComponent();
                        Stack<Point> stack = new Stack<Point>();
                        stack.Push(new Point(x, y));
                        visited[x, y] = true;
                        int minX = x, maxX = x, minY = y, maxY = y;

                        while (stack.Count > 0)
                        {
                            Point p = stack.Pop();
                            component.Pixels.Add(p);
                            if (p.X < minX) minX = p.X; if (p.X > maxX) maxX = p.X;
                            if (p.Y < minY) minY = p.Y; if (p.Y > maxY) maxY = p.Y;

                            for (int ny = p.Y - 1; ny <= p.Y + 1; ny++)
                                for (int nx = p.X - 1; nx <= p.X + 1; nx++)
                                    if (nx >= 0 && nx < w && ny >= 0 && ny < h &&
                                        !visited[nx, ny] && IsBlack(bmp.GetPixel(nx, ny)))
                                    {
                                        visited[nx, ny] = true;
                                        stack.Push(new Point(nx, ny));
                                    }
                        }
                        component.Rect = new Rectangle(minX, minY, maxX - minX + 1, maxY - minY + 1);
                        components.Add(component);
                    }
                }
            return components;
        }

        public static int[] GetHorizontalProjection(Bitmap bmp)
        {
            int[] projection = new int[bmp.Height];
            for (int y = 0; y < bmp.Height; y++)
                for (int x = 0; x < bmp.Width; x++)
                    if (IsBlack(bmp.GetPixel(x, y))) projection[y]++;
            return projection;
        }

        public static List<(int start, int end)> GetSegments(int[] projection, int threshold = 0)
        {
            List<(int, int)> segments = new();
            bool inSegment = false; int start = 0;
            for (int i = 0; i < projection.Length; i++)
            {
                if (!inSegment && projection[i] > threshold) { inSegment = true; start = i; }
                else if (inSegment && projection[i] <= threshold) { inSegment = false; segments.Add((start, i - 1)); }
            }
            if (inSegment) segments.Add((start, projection.Length - 1));
            return segments;
        }

        static List<ConnectedComponent> MergeDotComponents(List<ConnectedComponent> components, int lineHeight)
        {
            var result = new List<ConnectedComponent>();
            var used = new bool[components.Count];
            for (int i = 0; i < components.Count; i++)
            {
                if (used[i]) continue;
                var main = components[i];
                for (int j = i + 1; j < components.Count; j++)
                {
                    if (used[j]) continue;
                    var other = components[j];
                    bool overlapX = main.Rect.Left < other.Rect.Right && main.Rect.Right > other.Rect.Left;
                    int vertGap = Math.Max(0, Math.Max(main.Rect.Top, other.Rect.Top) - Math.Min(main.Rect.Bottom, other.Rect.Bottom));
                    bool closeY = vertGap < lineHeight * 0.4;
                    bool oneDot = other.Pixels.Count < main.Pixels.Count * 0.3 || main.Pixels.Count < other.Pixels.Count * 0.3;

                    if (overlapX && closeY && oneDot)
                    {
                        int x = Math.Min(main.Rect.X, other.Rect.X);
                        int y = Math.Min(main.Rect.Y, other.Rect.Y);
                        int r = Math.Max(main.Rect.Right, other.Rect.Right);
                        int b = Math.Max(main.Rect.Bottom, other.Rect.Bottom);
                        main.Rect = new Rectangle(x, y, r - x, b - y);
                        main.Pixels.AddRange(other.Pixels);
                        used[j] = true;
                    }
                }
                result.Add(main);
                used[i] = true;
            }
            return result;
        }


        static bool IsBlack(Color pixel) => pixel.R < 128;

        static void SaveDebugImages(Bitmap charBmp, string debugDir, int index, char predicted)
        {
            string rawPath = Path.Combine(debugDir, $"{index:D4}_raw_pred-{predicted}.png");
            charBmp.Save(rawPath);

            using (Bitmap processed = ImagePreprocessing.PreprocessImage(charBmp))
            using (Bitmap bigProcessed = new Bitmap(112, 112))
            using (Graphics gDbg = Graphics.FromImage(bigProcessed))
            {
                gDbg.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                gDbg.DrawImage(processed, 0, 0, 112, 112);
                string procPath = Path.Combine(debugDir, $"{index:D4}_processed_pred-{predicted}.png");
                bigProcessed.Save(procPath);
            }
        }

        public class ConnectedComponent
        {
            public Rectangle Rect;
            public List<Point> Pixels = new List<Point>();
        }
    }
}