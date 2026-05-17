using Vionet.VisionEngine;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Text;

namespace Vionet
{
    public static class ImageHelper
    {
        public static void SaveSampleFrom2DArray(
            float[,] data,
            int rowIndex,
            int width,
            int height,
            string filePath,
            int scale = 4)
        {
            if (data.GetLength(1) != width * height)
                throw new ArgumentException("Розмірність масиву не відповідає W * H");

            using Bitmap bmp = new Bitmap(width, height, PixelFormat.Format32bppRgb);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int colIndex = y * width + x;

                    float value = data[rowIndex, colIndex];
                    value = Math.Clamp(value, 0f, 1f);

                    int gray = 255 - (int)(value * 255);

                    bmp.SetPixel(x, y, Color.FromArgb(gray, gray, gray));
                }
            }

            using Bitmap scaled = new Bitmap(width * scale, height * scale);

            using (Graphics g = Graphics.FromImage(scaled))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Half;

                g.DrawImage(
                    bmp,
                    new Rectangle(0, 0, scaled.Width, scaled.Height),
                    new Rectangle(0, 0, bmp.Width, bmp.Height),
                    GraphicsUnit.Pixel
                );
            }

            scaled.Save(filePath, ImageFormat.Png);
        }

        static public void SaveDebugImages(Bitmap charBmp, string debugDir, int index, char predicted)
        {

            string rawPath = Path.Combine(debugDir, $"{index:D4}_raw_pred-{(int)predicted}.png");
            charBmp.Save(rawPath);

            using (Bitmap processed = ImagePreprocessing.PreprocessImage(charBmp))
            using (Bitmap bigProcessed = new Bitmap(112, 112))
            using (Graphics gDbg = Graphics.FromImage(bigProcessed))
            {
                gDbg.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                gDbg.DrawImage(processed, 0, 0, 112, 112);
                string procPath = Path.Combine(debugDir, $"{index:D4}_processed_pred-{(int)predicted}.png");
                bigProcessed.Save(procPath);
            }
        }
    }
}
