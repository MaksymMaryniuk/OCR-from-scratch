using Model;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;

namespace Model;

public static class DatasetGenerator
{
    public static (float[,] X, int[] y) GeneratePrintedData(
        string[] fontNames, string labels, int samplesPerChar = 50)
    {
        FontStyle[] styles = { FontStyle.Regular, FontStyle.Bold, FontStyle.Italic };
        int totalSamples = fontNames.Length * styles.Length * labels.Length * samplesPerChar;

        float[,] resX = new float[totalSamples, 784];
        int[] targets = new int[totalSamples];
        int sampleIndex = 0;

        Random rand = new Random();

        foreach (string fontName in fontNames)
        {
            foreach (var style in styles)
            {
                Font[] fonts = new Font[6];
                for (int i = 0; i < 6; i++)
                    fonts[i] = new Font(fontName, 26 + i, style);

                for (int i = 0; i < labels.Length; i++)
                {
                    string charStr = labels[i].ToString();

                    for (int s = 0; s < samplesPerChar; s++)
                    {
                        using (Bitmap tempBmp = new Bitmap(60, 60))
                        using (Graphics g = Graphics.FromImage(tempBmp))
                        {
                            g.TextRenderingHint = TextRenderingHint.AntiAlias;
                            g.SmoothingMode = SmoothingMode.AntiAlias;
                            g.Clear(Color.White);

                            Font currentFont = fonts[rand.Next(fonts.Length)];
                            float x = 10 + rand.Next(-5, 6);
                            float y = 10 + rand.Next(-5, 6);

                            float angle = (float)(rand.NextDouble() * 16 - 8);
                            g.TranslateTransform(30, 30);
                            g.RotateTransform(angle);
                            g.TranslateTransform(-30, -30);
                            g.DrawString(charStr, currentFont, Brushes.Black, x, y);
                            g.ResetTransform();


                            DataAugmentation.ApplyBlur(tempBmp);
                            DataAugmentation.ApplySaltPepper(tempBmp, intensity: 0.08);
                            DataAugmentation.ApplyBrightness(tempBmp, range: 30);

                            using (Bitmap processed = ImagePreprocessing.PreprocessImage(tempBmp))
                            {
                                float[] pixels = ImagePreprocessing.BitmapToArray(processed);
                                for (int p = 0; p < 784; p++)
                                    resX[sampleIndex, p] = pixels[p];
                            }
                        }

                        targets[sampleIndex] = i;
                        sampleIndex++;
                    }
                }

                foreach (var f in fonts) f.Dispose();
            }
        }

        return (resX, targets);
    }
}