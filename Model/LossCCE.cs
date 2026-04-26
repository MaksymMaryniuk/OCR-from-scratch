using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public class LossCCE : Loss
    {
        public float[] Outputs { get; set; }

        public override float[] Forward(float[,] y_pred, float[,] y_true)
        {
            //One-hot encoded version of CCE
            float epsilon = 1e-7F;

            for (int i = 0; i < y_pred.GetLength(0); i++)
            {
                for (int j = 0; j < y_pred.GetLength(1); j++)
                {
                    y_pred[i, j] = Math.Clamp(y_pred[i, j], epsilon, 1 - epsilon);
                }
            }

            float[] sampleLosses = new float[y_pred.GetLength(0)];

            for (int i = 0; i < y_pred.GetLength(0); i++)
            {
                for (int j = 0; j < y_true.GetLength(1); j++)
                {
                    if (y_true[i, j] == 1)
                    {
                        float correct_confidence = y_pred[i, j];
                        sampleLosses[i] = -MathF.Log(correct_confidence);
                        break;
                    }
                }

            }
            return sampleLosses;
        }
        public override float[] Forward(float[,] y_pred, int[] y_true)
        {
            //Scalar version of CCE
            float epsilon = 1e-7F;
            for (int i = 0; i < y_pred.GetLength(0); i++)
            {
                for (int j = 0; j < y_pred.GetLength(1); j++)
                {
                    y_pred[i, j] = Math.Clamp(y_pred[i, j], epsilon, 1 - epsilon);
                }
            }
            float[] sampleLosses = new float[y_pred.GetLength(0)];

            for (int i = 0; i < y_pred.GetLength(0); i++)
            {
                int correct_class_index = y_true[i];
                float correct_confidence = y_pred[i, correct_class_index];
                sampleLosses[i] = -MathF.Log(correct_confidence);
            }
            return sampleLosses;
        }
        public override float[,] Backward(float[,] softmax_output,  int[] y_true)
        {
            
            float[,] output = new float[softmax_output.GetLength(0), softmax_output.GetLength(1)];
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