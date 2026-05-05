using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;

namespace Model
{
    public static class ImagePreprocessing
    {
        static Random rand = new Random();


        public static float[] BitmapToArray(Bitmap bmp)
        {
            float[] data = new float[bmp.Width * bmp.Height];
            for (int y = 0; y < bmp.Height; y++)
                for (int x = 0; x < bmp.Width; x++)
                {
                    data[y * bmp.Width + x] = 1.0F - (bmp.GetPixel(x, y).R / 255.0F);
                }
            return data;
        }

        public static float[,] GetInputForModel(Bitmap charBmp)
        {
            using (Bitmap processed = PreprocessImage(charBmp))
            {
                float[] pixels = BitmapToArray(processed);
                float[,] inputMatrix = new float[1, 784];
                for (int i = 0; i < 784; i++) inputMatrix[0, i] = pixels[i];
                return inputMatrix;
            }
        }

        public static Bitmap PreprocessImage(Bitmap bmp)
        {
            Bitmap gray = ToGrayscale(bmp);
            Rectangle bbox = FindBoundingBox(gray);
            if (bbox.Width == 0 || bbox.Height == 0)
            {
                gray.Dispose();
                return new Bitmap(28, 28);
            }

            Bitmap cropped = gray.Clone(bbox, gray.PixelFormat);
            gray.Dispose();

            Bitmap resized = ResizeKeepAspect(cropped, 16, 16);
            cropped.Dispose();

            Bitmap canvas = PlaceInCenter(resized, 28, 28);
            resized.Dispose();

            Bitmap centered = CenterImage(canvas);
            canvas.Dispose();

            return centered;
        }

        public static Bitmap ToGrayscale(Bitmap bmp)
        {
            Bitmap gray = new Bitmap(bmp.Width, bmp.Height);
            for (int y = 0; y < bmp.Height; y++)
                for (int x = 0; x < bmp.Width; x++)
                {
                    Color c = bmp.GetPixel(x, y);
                    int g = (c.R + c.G + c.B) / 3;
                    gray.SetPixel(x, y, Color.FromArgb(g, g, g));
                }
            return gray;
        }

        public static Bitmap Threshold(Bitmap bmp, int threshold)
        {
            Bitmap binary = new Bitmap(bmp.Width, bmp.Height);
            for (int y = 0; y < bmp.Height; y++)
                for (int x = 0; x < bmp.Width; x++)
                {
                    int val = bmp.GetPixel(x, y).R;
                    binary.SetPixel(x, y, val < threshold ? Color.Black : Color.White);
                }
            return binary;
        }

        public static int GetOtsuThreshold(Bitmap bmp)
        {
            int[] histogram = new int[256];
            for (int y = 0; y < bmp.Height; y++)
                for (int x = 0; x < bmp.Width; x++)
                    histogram[bmp.GetPixel(x, y).R]++;

            float sum = 0;
            for (int i = 0; i < 256; i++) sum += i * histogram[i];

            float sumB = 0;
            int wB = 0, wF = 0;
            float varMax = 0;
            int threshold = 0;

            for (int i = 0; i < 256; i++)
            {
                wB += histogram[i];
                if (wB == 0) continue;
                wF = (bmp.Width * bmp.Height) - wB;
                if (wF == 0) break;

                sumB += (float)(i * histogram[i]);
                float mB = sumB / wB;
                float mF = (sum - sumB) / wF;
                float varBetween = (float)wB * wF * (mB - mF) * (mB - mF);

                if (varBetween > varMax)
                {
                    varMax = varBetween;
                    threshold = i;
                }
            }
            return threshold;
        }

        public static Bitmap ScaleImageForOCR(Bitmap bmp, int maxWidth = 1200)
        {
            if (bmp.Width <= maxWidth) return new Bitmap(bmp);
            float ratio = (float)maxWidth / bmp.Width;
            Bitmap resized = new Bitmap(maxWidth, (int)(bmp.Height * ratio));
            using (Graphics g = Graphics.FromImage(resized))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.DrawImage(bmp, 0, 0, resized.Width, resized.Height);
            }
            return resized;
        }

        static Rectangle FindBoundingBox(Bitmap bmp)
        {
            int minX = bmp.Width, minY = bmp.Height, maxX = 0, maxY = 0;
            bool found = false;
            for (int y = 0; y < bmp.Height; y++)
                for (int x = 0; x < bmp.Width; x++)
                    if (bmp.GetPixel(x, y).R < 128)
                    {
                        if (x < minX) minX = x; if (y < minY) minY = y;
                        if (x > maxX) maxX = x; if (y > maxY) maxY = y;
                        found = true;
                    }
            return found ? new Rectangle(minX, minY, maxX - minX + 1, maxY - minY + 1) : Rectangle.Empty;
        }

        static Bitmap ResizeKeepAspect(Bitmap bmp, int maxW, int maxH)
        {
            float scale = Math.Min((float)maxW / bmp.Width, (float)maxH / bmp.Height);
            int newW = Math.Max(1, (int)(bmp.Width * scale));
            int newH = Math.Max(1, (int)(bmp.Height * scale));
            Bitmap result = new Bitmap(newW, newH);
            using (Graphics g = Graphics.FromImage(result))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.Clear(Color.White);
                g.DrawImage(bmp, 0, 0, newW, newH);
            }
            return result;
        }

        static Bitmap PlaceInCenter(Bitmap bmp, int width, int height)
        {
            Bitmap result = new Bitmap(width, height);
            using (Graphics g = Graphics.FromImage(result))
            {
                g.Clear(Color.White);
                g.DrawImage(bmp, (width - bmp.Width) / 2, (height - bmp.Height) / 2);
            }
            return result;
        }

        static Bitmap CenterImage(Bitmap bmp)
        {
            var (cx, cy) = GetCenterOfMass(bmp);
            Bitmap result = new Bitmap(bmp.Width, bmp.Height);
            using (Graphics g = Graphics.FromImage(result))
            {
                g.Clear(Color.White);
                g.DrawImage(bmp, (int)(bmp.Width / 2 - cx), (int)(bmp.Height / 2 - cy));
            }
            return result;
        }

        static (double cx, double cy) GetCenterOfMass(Bitmap bmp)
        {
            double sumX = 0, sumY = 0, total = 0;
            for (int y = 0; y < bmp.Height; y++)
                for (int x = 0; x < bmp.Width; x++)
                {
                    double weight = 1.0 - (bmp.GetPixel(x, y).R / 255.0);
                    sumX += x * weight; sumY += y * weight; total += weight;
                }
            return total == 0 ? (bmp.Width / 2.0, bmp.Height / 2.0) : (sumX / total, sumY / total);
        }
    }
}