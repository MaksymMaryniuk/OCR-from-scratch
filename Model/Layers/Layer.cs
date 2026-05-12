namespace Model;

public abstract class Layer
{
    public abstract string Type { get; }

    public virtual bool IsTrainable => true;
    public float[,] Inputs { get; set; }
    public float[,] Output { get; set; }
    public float[,] Dinputs { get; set; }

    public abstract void Forward(float[,] inputs);
    public abstract void Backward(float[,] dvalues);
}
