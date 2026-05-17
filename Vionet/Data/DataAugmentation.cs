using Vionet.VisionEngine;
using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace Vionet.Data;

public static class DataAugmentation
{
    private static readonly Random Rng = new Random();
    public static float[] AugmentArray(float[] image, int maxShift = 2, double noiseLevel = 0.05)
    {
        float[] result = new float[784];

        int dx = Rng.Next(-maxShift, maxShift + 1);
        int dy = Rng.Next(-maxShift, maxShift + 1);

        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++)
            {
                int sx = x - dx, sy = y - dy;
                result[y * 28 + x] = (sx >= 0 && sx < 28 && sy >= 0 && sy < 28)
                    ? image[sy * 28 + sx] : 0f;
            }

        ApplySaltPepperArray(result, noiseLevel);
        return result;
    }
    public static float[,] AugmentBatch(float[,] source, int maxShift = 2, double noiseLevel = 0.05)
    {
        int rows = source.GetLength(0);
        int cols = source.GetLength(1);
        float[,] result = new float[rows, cols];

        int dx = Rng.Next(-maxShift, maxShift + 1);
        int dy = Rng.Next(-maxShift, maxShift + 1);

        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
            {
                int oy = y - dy, ox = x - dx;
                result[y, x] = (ox >= 0 && ox < cols && oy >= 0 && oy < rows)
                    ? source[oy, ox] : 0f;
            }

        ApplyGaussianNoiseArray(result, noiseLevel);
        return result;
    }

    public static void ApplyBlur(Bitmap bmp, double chance = 0.4)
    {
        if (Rng.NextDouble() > chance) return;

        int w = bmp.Width, h = bmp.Height;
        Color[,] copy = new Color[w, h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                copy[x, y] = bmp.GetPixel(x, y);

        for (int y = 1; y < h - 1; y++)
            for (int x = 1; x < w - 1; x++)
            {
                int r = 0, count = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                    { r += copy[x + dx, y + dy].R; count++; }
                int avg = r / count;
                bmp.SetPixel(x, y, Color.FromArgb(avg, avg, avg));
            }
    }

    public static void ApplySaltPepper(Bitmap bmp, double intensity = 0.005)
    {
        int noisePixels = (int)(bmp.Width * bmp.Height * intensity);

        for (int n = 0; n < noisePixels; n++)
        {
            int x = Rng.Next(bmp.Width);
            int y = Rng.Next(bmp.Height);

            Color current = bmp.GetPixel(x, y);

            if (current.R < 180)
                continue;

            int val = Rng.NextDouble() < 0.5 ? 255 : 0;

            bmp.SetPixel(x, y, Color.FromArgb(val, val, val));
        }
    }

    public static void ApplyBrightness(Bitmap bmp, int range = 30)
    {
        int shift = Rng.Next(-range, range + 1);
        if (Math.Abs(shift) < 5) return;

        for (int y = 0; y < bmp.Height; y++)
            for (int x = 0; x < bmp.Width; x++)
            {
                int val = Math.Clamp(bmp.GetPixel(x, y).R + shift, 0, 255);
                bmp.SetPixel(x, y, Color.FromArgb(val, val, val));
            }
    }

    static void ApplySaltPepperArray(float[] arr, double noiseLevel)
    {
        for (int i = 0; i < arr.Length; i++)
            if (Rng.NextDouble() < noiseLevel)
                arr[i] = Rng.NextDouble() > 0.5 ? 1f : 0f;
    }

    static void ApplyGaussianNoiseArray(float[,] arr, double noiseLevel)
    {
        for (int i = 0; i < arr.GetLength(0); i++)
            for (int j = 0; j < arr.GetLength(1); j++)
                arr[i, j] = Math.Clamp(
                    arr[i, j] + (float)((Rng.NextDouble() * 2 - 1) * noiseLevel), 0f, 1f);
    }

    public static void ApplyAugmentation(Bitmap bmp, AugmentationConfig aug)
    {

        if (aug.UseBrightness)
            ApplyBrightness(bmp, aug.BrightnessRange);

        if (aug.UseSaltPepper)
            ApplySaltPepper(bmp, aug.SaltPepperIntensity);

        if (aug.UseBlur)
            ApplyBlur(bmp, aug.BlurChance);
    }
    public static float[,] AugmentExistingDataset(float[,] X, AugmentationConfig aug)
    {
        int samples = X.GetLength(0);
        float[,] augmentedX = new float[samples, 784];

        for (int i = 0; i < samples; i++)
        {
            float[] sample = new float[784];
            for (int p = 0; p < 784; p++) sample[p] = X[i, p];

            using (Bitmap bmp = ImagePreprocessing.ArrayToBitmap(sample, 28, 28))
            {
                DataAugmentation.ApplyAugmentation(bmp, aug);

                float[] processedPixels = ImagePreprocessing.BitmapToArray(bmp);
                for (int p = 0; p < 784; p++)
                    augmentedX[i, p] = processedPixels[p];
            }
        }

        return augmentedX;
    }
}
public class AugmentationConfig
{
    public bool UseBlur { get; set; } = false;
    public double BlurChance { get; set; } = 0.4;

    public bool UseSaltPepper { get; set; } = true;
    public double SaltPepperIntensity { get; set; } = 0.05;

    public bool UseBrightness { get; set; } = false;
    public int BrightnessRange { get; set; } = 20;

    public bool UseShift { get; set; } = true;
    public int MaxShift { get; set; } = 2;
}