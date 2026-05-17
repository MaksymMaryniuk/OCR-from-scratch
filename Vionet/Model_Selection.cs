using System;
using System.Collections.Generic;
using System.Text;

namespace Vionet
{
    public class ModelSelection
    {
        public static (float[,] X_train, float[,] X_val, int[] y_train, int[] y_val)
            TrainValSplit(float[,] X, int[] y, float valSplit = 0.2f)
        {
            int total = X.GetLength(0);
            int features = X.GetLength(1);
            int valSize = (int)(total * valSplit);
            int trainSize = total - valSize;

            float[,] X_train = new float[trainSize, features];
            float[,] X_val = new float[valSize, features];
            int[] y_train = new int[trainSize];
            int[] y_val = new int[valSize];

            for (int i = 0; i < trainSize; i++)
            {
                y_train[i] = y[i];
                for (int j = 0; j < features; j++) X_train[i, j] = X[i, j];
            }
            for (int i = 0; i < valSize; i++)
            {
                y_val[i] = y[trainSize + i];
                for (int j = 0; j < features; j++) X_val[i, j] = X[trainSize + i, j];
            }

            return (X_train, X_val, y_train, y_val);
        }

        public static double Validate(
            Model model,
            float[,] X_val, int[] y_val,
            string labels = null,
            double weakThreshold = 80.0)
        {
            int total = X_val.GetLength(0);
            int features = X_val.GetLength(1);
            int correct = 0;

            int[] perClassCorrect = labels != null ? new int[labels.Length] : null;
            int[] perClassTotal = labels != null ? new int[labels.Length] : null;

            for (int i = 0; i < total; i++)
            {
                float[,] input = new float[1, features];
                for (int j = 0; j < features; j++) input[0, j] = X_val[i, j];

                var output = model.Forward(input);
                int pred = AdditionalMath.GetArgmax(output);

                if (pred == y_val[i]) correct++;

                if (labels != null && y_val[i] < labels.Length)
                {
                    perClassTotal[y_val[i]]++;
                    if (pred == y_val[i]) perClassCorrect[y_val[i]]++;
                }
            }

            double accuracy = (double)correct / total * 100;
            Console.WriteLine($"Val accuracy: {accuracy:F2}%  ({correct}/{total})");

            if (labels != null)
            {
                Console.WriteLine($"\nWeak classes (< {weakThreshold}%):");
                bool anyWeak = false;
                for (int c = 0; c < labels.Length; c++)
                {
                    if (perClassTotal[c] == 0) continue;
                    double acc = (double)perClassCorrect[c] / perClassTotal[c] * 100;
                    if (acc < weakThreshold)
                    {
                        Console.WriteLine($"  '{labels[c]}' : {acc:F1}%  ({perClassCorrect[c]}/{perClassTotal[c]})");
                        anyWeak = true;
                    }
                }
                if (!anyWeak) Console.WriteLine("All classes above threshold.");
            }

            return accuracy;
        }
    }
}
