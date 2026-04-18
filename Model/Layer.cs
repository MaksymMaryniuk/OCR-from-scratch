namespace Model;

public abstract class Layer
{
    public double[,] Inputs { get; set; }
    public double[,] Output { get; set; }
    public double[,] Dinputs { get; set; }
    public Layer Prev { get; set; }
    public Layer Next { get; set; }

    public abstract void Forward(double[,] inputs);
    public abstract void Backward(double[,] dvalues);
}
