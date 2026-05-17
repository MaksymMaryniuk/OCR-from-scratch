using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;

namespace Vionet.VisionEngine
{
    public static class Segmentation
    {
        // =====================================================================
        // ПУБЛІЧНІ МЕТОДИ РОЗПІЗНАВАННЯ
        // =====================================================================

        public static string RecognizeTextFromPhoto(
            string imagePath, Model nn, string labels, string debugDir = null)
        {
            using (Bitmap original = new Bitmap(imagePath))
            {
                (string text, float[] confidence) =
                    RecognizeTextFromBitmap(original, nn, labels, out _, debugDir);
                return text;
            }
        }

        public static (string text, float[] confidence) RecognizeTextFromBitmap(
            Bitmap original, Model nn, string labels,
            out List<Rectangle> boundingBoxes, string debugDir = null)
        {
            boundingBoxes = new List<Rectangle>();
            var confidenceList = new List<float>();
            StringBuilder fullText = new StringBuilder();

            PrepareDebugDir(debugDir);
            int debugIndex = 0;

            using (Bitmap scaled = ImagePreprocessing.ScaleImageForOCR(original))
            using (Bitmap gray = ImagePreprocessing.ToGrayscale(scaled))
            {
                float scaleX = (float)original.Width / scaled.Width;
                float scaleY = (float)original.Height / scaled.Height;

                int optimalThreshold = ImagePreprocessing.GetOtsuThreshold(gray);
                using (Bitmap binary = ImagePreprocessing.Threshold(gray, optimalThreshold))
                {
                    foreach (var line in GetLineSegments(binary))
                    {
                        ProcessLine(
                            line, binary, gray,
                            nn, labels,
                            scaleX, scaleY,
                            fullText, confidenceList, boundingBoxes,
                            debugDir, ref debugIndex);
                    }
                }
            }

            string rawText = fullText.ToString().TrimEnd();
            string formattedText = ApplySentenceCasing(rawText);

            return (formattedText, confidenceList.ToArray());
        
        }

        // =====================================================================
        // ОБРОБКА ОДНОГО РЯДКА
        // =====================================================================

        static void ProcessLine(
            (int start, int end) line,
            Bitmap binary, Bitmap gray,
            Model nn, string labels,
            float scaleX, float scaleY,
            StringBuilder fullText, List<float> confidenceList,
            List<Rectangle> boundingBoxes,
            string debugDir, ref int debugIndex)
        {
            int lineHeight = line.end - line.start + 1;
            if (lineHeight < 10) return;

            Rectangle lineRect = GetPaddedLineRect(line, binary.Width, binary.Height);

            using (Bitmap lineBinary = binary.Clone(lineRect, binary.PixelFormat))
            using (Bitmap lineGray = gray.Clone(lineRect, gray.PixelFormat))
            {
                var components = GetLineComponents(lineBinary, lineHeight);

                for (int i = 0; i < components.Count; i++)
                {
                    var comp = components[i];

                    AddBoundingBox(comp, lineRect, scaleX, scaleY, boundingBoxes);

                    RecognizeComponent(
                        comp, lineGray, nn, labels,
                        fullText, confidenceList,
                        debugDir, ref debugIndex);

                    if (i < components.Count - 1)
                        AppendSpaceIfNeeded(fullText, components[i], components[i + 1], lineHeight);
                }

                fullText.AppendLine();
            }
        }

        // =====================================================================
        // СЕГМЕНТАЦІЯ РЯДКІВ І КОМПОНЕНТІВ
        // =====================================================================

        static IEnumerable<(int start, int end)> GetLineSegments(Bitmap binary)
        {
            var hProjection = GetHorizontalProjection(binary);
            return GetSegments(hProjection, threshold: 2);
        }

        static Rectangle GetPaddedLineRect(
            (int start, int end) line, int bmpWidth, int bmpHeight)
        {
            int lineHeight = line.end - line.start + 1;
            int ascenderPad = 3;
            int descenderPad = 8;

            int paddedStart = Math.Max(0, line.start - ascenderPad);
            int paddedEnd = Math.Min(bmpHeight - 1, line.end + descenderPad);
            int paddedHeight = paddedEnd - paddedStart + 1;

            return new Rectangle(0, paddedStart, bmpWidth, paddedHeight);
        }

        static List<ConnectedComponent> GetLineComponents(Bitmap lineBinary, int lineHeight)
        {
            var components = GetConnectedComponents(lineBinary)
                .Where(c => c.Rect.Width >= 1 && c.Rect.Height >= 1)
                .OrderBy(c => c.Rect.X)
                .ToList();

            components = MergeDotComponents(components, lineHeight);
            return components.OrderBy(c => c.Rect.X).ToList();
        }

        // =====================================================================
        // РОЗПІЗНАВАННЯ СИМВОЛУ
        // =====================================================================

        static void RecognizeComponent(
            ConnectedComponent comp, Bitmap lineGray,
            Model nn, string labels,
            StringBuilder fullText, List<float> confidenceList,
            string debugDir, ref int debugIndex)
        {
            using (Bitmap charBmp = lineGray.Clone(comp.Rect, lineGray.PixelFormat))
            {
                var input = ImagePreprocessing.GetInputForModel(charBmp);
                var output = nn.Forward(input);
                int idx = AdditionalMath.GetArgmax(output);

                confidenceList.Add(output[0, idx]);
                char predicted = labels[idx];
                fullText.Append(predicted);

                if (debugDir != null)
                {
                    ImageHelper.SaveDebugImages(charBmp, debugDir, debugIndex, predicted);
                    debugIndex++;
                }
            }
        }

        static void AddBoundingBox(
            ConnectedComponent comp, Rectangle lineRect,
            float scaleX, float scaleY,
            List<Rectangle> boundingBoxes)
        {
            Rectangle absRect = new Rectangle(
                comp.Rect.X + lineRect.X,
                comp.Rect.Y + lineRect.Y,
                comp.Rect.Width,
                comp.Rect.Height);

            boundingBoxes.Add(new Rectangle(
                (int)(absRect.X * scaleX), (int)(absRect.Y * scaleY),
                (int)(absRect.Width * scaleX), (int)(absRect.Height * scaleY)));
        }

        static void AppendSpaceIfNeeded(
            StringBuilder fullText,
            ConnectedComponent current, ConnectedComponent next,
            int lineHeight)
        {
            int gap = next.Rect.X - current.Rect.Right;
            if (gap > lineHeight * 0.25)
                fullText.Append(" ");
        }

        // =====================================================================
        // CONNECTED COMPONENTS
        // =====================================================================

        public static List<ConnectedComponent> GetConnectedComponents(Bitmap bmp)
        {
            int w = bmp.Width, h = bmp.Height;
            bool[,] visited = new bool[w, h];
            var components = new List<ConnectedComponent>();

            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    if (IsBlack(bmp.GetPixel(x, y)) && !visited[x, y])
                        components.Add(FloodFill(bmp, visited, x, y, w, h));

            return components;
        }

        static ConnectedComponent FloodFill(
            Bitmap bmp, bool[,] visited, int startX, int startY, int w, int h)
        {
            var component = new ConnectedComponent();
            var stack = new Stack<Point>();
            stack.Push(new Point(startX, startY));
            visited[startX, startY] = true;

            int minX = startX, maxX = startX;
            int minY = startY, maxY = startY;

            while (stack.Count > 0)
            {
                Point p = stack.Pop();
                component.Pixels.Add(p);

                if (p.X < minX) minX = p.X; if (p.X > maxX) maxX = p.X;
                if (p.Y < minY) minY = p.Y; if (p.Y > maxY) maxY = p.Y;

                for (int ny = p.Y - 1; ny <= p.Y + 1; ny++)
                    for (int nx = p.X - 1; nx <= p.X + 1; nx++)
                        if (nx >= 0 && nx < w && ny >= 0 && ny < h
                            && !visited[nx, ny] && IsBlack(bmp.GetPixel(nx, ny)))
                        {
                            visited[nx, ny] = true;
                            stack.Push(new Point(nx, ny));
                        }
            }

            component.Rect = new Rectangle(minX, minY, maxX - minX + 1, maxY - minY + 1);
            return component;
        }

        // =====================================================================
        // MERGE / ПРОЕКЦІЇ / ДОПОМІЖНЕ
        // =====================================================================

        static List<ConnectedComponent> MergeDotComponents(
            List<ConnectedComponent> components, int lineHeight)
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

                    bool overlapX = main.Rect.Left < other.Rect.Right &&
                                    main.Rect.Right > other.Rect.Left;
                    int vertGap = Math.Max(0,
                                        Math.Max(main.Rect.Top, other.Rect.Top) -
                                        Math.Min(main.Rect.Bottom, other.Rect.Bottom));
                    bool closeY = vertGap < lineHeight * 0.4;
                    bool oneDot = other.Pixels.Count < main.Pixels.Count * 0.3 ||
                                    main.Pixels.Count < other.Pixels.Count * 0.3;

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

        public static int[] GetHorizontalProjection(Bitmap bmp)
        {
            int[] projection = new int[bmp.Height];
            for (int y = 0; y < bmp.Height; y++)
                for (int x = 0; x < bmp.Width; x++)
                    if (IsBlack(bmp.GetPixel(x, y))) projection[y]++;
            return projection;
        }

        public static List<(int start, int end)> GetSegments(
            int[] projection, int threshold = 0)
        {
            var segments = new List<(int, int)>();
            bool inSegment = false;
            int start = 0;

            for (int i = 0; i < projection.Length; i++)
            {
                if (!inSegment && projection[i] > threshold)
                { inSegment = true; start = i; }
                else if (inSegment && projection[i] <= threshold)
                { inSegment = false; segments.Add((start, i - 1)); }
            }
            if (inSegment) segments.Add((start, projection.Length - 1));
            return segments;
        }

        static void PrepareDebugDir(string debugDir)
        {
            if (debugDir == null) return;
            if (Directory.Exists(debugDir)) Directory.Delete(debugDir, recursive: true);
            Directory.CreateDirectory(debugDir);
        }

        static bool IsBlack(Color pixel) => pixel.R < 128;

        // =====================================================================
        // МОДЕЛІ ДАНИХ
        // =====================================================================

        public class ConnectedComponent
        {
            public Rectangle Rect;
            public List<Point> Pixels = new List<Point>();
        }


        // =====================================================================
        // ПОСТОБРОБКА ТЕКСТУ
        // =====================================================================

        private static string ApplySentenceCasing(string input)
        {
            if (string.IsNullOrEmpty(input)) return input;

            char[] chars = input.ToCharArray();
            bool capitalizeNext = true; // Перша літера тексту має бути великою

            for (int i = 0; i < chars.Length; i++)
            {
                char c = chars[i];

                // Якщо це літера (ігнорує цифри, пробіли та пунктуацію)
                if (char.IsLetter(c))
                {
                    if (capitalizeNext)
                    {
                        chars[i] = char.ToUpper(c);
                        capitalizeNext = false; // Вимикаємо прапорець до наступної крапки
                    }
                    else
                    {
                        chars[i] = char.ToLower(c); // Всі інші літери примусово малі
                    }
                }
                // Якщо зустріли кінець речення, наступна літера буде великою
                else if (c == '.' || c == '!' || c == '?')
                {
                    capitalizeNext = true;
                }
                // Всі інші символи (пробіли, коми, цифри) просто пропускаються 
                // і залишаються як є, не змінюючи стан capitalizeNext
            }

            return new string(chars);
        }
    }
}