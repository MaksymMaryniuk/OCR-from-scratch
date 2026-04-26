using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    using System;

    using System;
    using System.Drawing;
    using System.Drawing.Imaging;

    public static class ImagePreprocessing
    {
        static Random rand = new Random();

        static public float[] BitmapToArray(Bitmap bmp)
        {
            int w = bmp.Width;
            int h = bmp.Height;
            float[] data = new float[w * h];

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    Color pixel = bmp.GetPixel(x, y);

                    float gray = (float)((pixel.R + pixel.G + pixel.B) / 3.0);

                    data[y * w + x] = 1.0F - (gray / 255.0F);
                }
            }

            return data;
        }
        public static float[] GetInputFromCanvas(Bitmap userDrawing)
        {
            Bitmap rescaled = new Bitmap(userDrawing, new Size(28, 28));

            float[] input = ImagePreprocessing.BitmapToArray(rescaled);

            return input;
        }

        public static float[,] GetInputForModel(Bitmap userDrawing)
        {
            float[] pixels = GetInputFromCanvas(userDrawing);

            float[,] inputMatrix = new float[1, 784];

            for (int i = 0; i < 784; i++)
            {
                inputMatrix[0, i] = pixels[i];
            }

            return inputMatrix;
        }
    }
}
