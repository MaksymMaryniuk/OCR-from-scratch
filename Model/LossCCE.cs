using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public class LossCCE : Loss
    {
        public double[] Outputs { get; set; }

        public override double[] Forward(double[,] y_pred, double[,] y_true)
        {
            //One-hot encoded version of CCE
            double epsilon = 1e-7;

            for (int i = 0; i < y_pred.GetLength(0); i++)
            {
                for (int j = 0; j < y_pred.GetLength(1); j++)
                {
                    y_pred[i, j] = Math.Clamp(y_pred[i, j], epsilon, 1 - epsilon);
                }
            }

            double[] sampleLosses = new double[y_pred.GetLength(0)];

            for (int i = 0; i < y_pred.GetLength(0); i++)
            {
                for (int j = 0; j < y_true.GetLength(1); j++)
                {
                    if (y_true[i, j] == 1)
                    {
                        double correct_confidence = y_pred[i, j];
                        sampleLosses[i] = -Math.Log(correct_confidence);
                        break;
                    }
                }

            }
            return sampleLosses;
        }
        public override double[] Forward(double[,] y_pred, int[] y_true)
        {
            //Scalar version of CCE
            double epsilon = 1e-7;
            for (int i = 0; i < y_pred.GetLength(0); i++)
            {
                for (int j = 0; j < y_pred.GetLength(1); j++)
                {
                    y_pred[i, j] = Math.Clamp(y_pred[i, j], epsilon, 1 - epsilon);
                }
            }
            double[] sampleLosses = new double[y_pred.GetLength(0)];

            for (int i = 0; i < y_pred.GetLength(0); i++)
            {
                int correct_class_index = y_true[i];
                double correct_confidence = y_pred[i, correct_class_index];
                sampleLosses[i] = -Math.Log(correct_confidence);
            }
            return sampleLosses;
        }
        public override double[,] Backward(double[,] softmax_output,  int[] y_true)
        {
            
            double[,] output = new double[softmax_output.GetLength(0), softmax_output.GetLength(1)];
            for (int i = 0; i < softmax_output.GetLength(0); i++)
            {
                for (int j = 0; j < softmax_output.GetLength(1); j++)
                {
                    output[i, j] = softmax_output[i, j] - (y_true[i] == j ? 1 : 0);
                }
            }
            return output;
        }
    }
}