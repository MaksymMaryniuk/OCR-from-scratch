namespace Model;

public abstract class Layer
{
    public double[,] Inputs { get; set; }
    public double[,] Output { get; set; }

    public abstract void Forward(double[,] inputs);
    public abstract double[,] Backward(double[,] dvalues);
}
