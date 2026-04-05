using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    using System;

    public class DataGenerator
    {
        // Generates a dataset of points arranged in a spiral pattern for a specified number of classes and points per class.
        public static (double[,] X, int[] y) CreateData(int points, int classes)
        {
            double[,] X = new double[points * classes, 2];
            int[] y = new int[points * classes];
            Random rand = new Random(0);

            for (int classNumber = 0; classNumber < classes; classNumber++)
            {
                for (int i = 0; i < points; i++)
                {
                    int ix = points * classNumber + i;

                    double r = (double)i / (points - 1);

                    double tStart = classNumber * 4;
                    double tEnd = (classNumber + 1) * 4;
                    double t = tStart + ((double)i / (points - 1)) * (tEnd - tStart);

                    double u1 = 1.0 - rand.NextDouble();
                    double u2 = 1.0 - rand.NextDouble();
                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                    t += randStdNormal * 0.2;
                    X[ix, 0] = r * Math.Sin(t * 2.5);
                    X[ix, 1] = r * Math.Cos(t * 2.5);
                    y[ix] = classNumber;
                }
            }
            return (X, y);
        }
    }
}
