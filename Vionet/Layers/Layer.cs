namespace Vionet.Layers;

public abstract class Layer
{
    public abstract string Type { get; }

    public virtual bool IsTrainable => true;

    public bool IsTraining { get; set; } = true;

    public float[,] Inputs { get; protected set; }
    public float[,] Output { get; protected set; }
    public float[,] Dinputs { get; protected set; }

    internal abstract void Forward(float[,] inputs);
    internal abstract void Backward(float[,] dvalues);
}