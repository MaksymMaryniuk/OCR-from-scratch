namespace Model;

public abstract class Layer
{
    public float[,] Inputs { get; set; }
    public float[,] Output { get; set; }
    public float[,] Dinputs { get; set; }
    public Layer Prev { get; set; }
    public Layer Next { get; set; }

    public abstract void Forward(float[,] inputs);
    public abstract void Backward(float[,] dvalues);
}
