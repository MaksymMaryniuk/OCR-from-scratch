namespace Model;

public class Layer
{
    public double[,] Weights { get; set; }
    public double[] Biases { get; set; }
    public double[,] Output { get; set; }
    private Random rand = new Random();
    public Layer(int num_inputs, int num_neurons)
    {
        Weights = new double[num_inputs, num_neurons];
        Biases = new double[num_neurons];
        Output = new double[num_inputs, num_neurons];
        Matrix_filler(Weights);
        Vector_filler(Biases);
    }
    public void Forward(double[,] X)
    {
        Output = Matrix_Multiplier(X, Weights, Biases);
    }


    private void Matrix_filler(double[,] value)
    {

        for (int i = 0; i < value.GetLength(0); i++)
        {
            for (int j = 0; j < value.GetLength(1); j++)
            {
                value[i, j] = (rand.NextDouble() * 2 - 1) * 0.1;
            }
        }
    }
    private void Vector_filler(double[] value)
    {
        for (int i = 0; i < value.Length; i++)
        {
            value[i] = rand.NextDouble() * 2 - 1;
        }
    }

    private double[,] Matrix_Multiplier(double[,] matrix1, double[,] matrix2, double[] bias)
    {
        int output_rows = matrix1.GetLength(0);
        int output_cols = matrix2.GetLength(1);
        double[,] output = new double[output_rows, output_cols];

        for (int rows = 0; rows < output_rows; rows++)
        {
            for (int cols = 0; cols < output_cols; cols++)
            {
                double sum = 0;
                for (int common = 0; common < matrix1.GetLength(1); common++)
                {
                    sum += matrix1[rows, common] * matrix2[common, cols];
                }
                output[rows, cols] = sum + bias[cols];
            }
        }
        return output;
    }
}

