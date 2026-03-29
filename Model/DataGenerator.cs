using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    using System;

    public class DataGenerator
    {
        public static (double[,] X, byte[] y) CreateData(int points, int classes)
        {
            double[,] X = new double[points * classes, 2];
            byte[] y = new byte[points * classes];
            Random rand = new Random(0); // seed(0)

            for (int classNumber = 0; classNumber < classes; classNumber++)
            {
                for (int i = 0; i < points; i++)
                {
                    int ix = points * classNumber + i;

                    // Аналог np.linspace(0.0, 1, points)
                    double r = (double)i / (points - 1);

                    // Аналог np.linspace для кута t + додавання шуму randn
                    double tStart = classNumber * 4;
                    double tEnd = (classNumber + 1) * 4;
                    double t = tStart + ((double)i / (points - 1)) * (tEnd - tStart);

                    // Додаємо шум (Box-Muller transform для нормального розподілу randn)
                    double u1 = 1.0 - rand.NextDouble();
                    double u2 = 1.0 - rand.NextDouble();
                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                    t += randStdNormal * 0.2;

                    // X[ix] = [r*sin(t*2.5), r*cos(t*2.5)]
                    X[ix, 0] = r * Math.Sin(t * 2.5);
                    X[ix, 1] = r * Math.Cos(t * 2.5);
                    y[ix] = (byte)classNumber;
                }
            }
            return (X, y);
        }
    }
}
