using Model;
using Model.Layers;
using Model.Optimizers;

namespace Optical_Recognition_Pattern
{
    public partial class Form1 : Form
    {
        Bitmap userDrawing = new Bitmap(280, 280);
        Graphics g;
        private Model.Model nn;
        private string emnistLabels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";

        public Form1()
        {
            InitializeComponent();
            g = Graphics.FromImage(userDrawing);
            g.Clear(Color.White);
            pictureBox1.Image = userDrawing;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            InitializeModel();
        }

        private void pictureBox1_MouseMove_1(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                g.FillEllipse(Brushes.Black, e.X, e.Y, 15, 15);
                pictureBox1.Invalidate();
            }
        }

        private void btnPredict_Click_1(object sender, EventArgs e)
        {
            if (nn == null) return;

            float[,] input = ImagePreprocessing.GetInputForModel(userDrawing);

            float[,] output = nn.Forward(input);

            int predictedIndex = AdditionalMath.GetArgmax(output);

            char result = emnistLabels[predictedIndex];
            lblResult.Text = $"Це символ: {result}";
            //lblConfidence.Text = $"Впевненість: {output[0, predictedIndex] * 100:F2}%";
        }

        private void btnClear_Click_1(object sender, EventArgs e)
        {
            g.Clear(Color.White);
            pictureBox1.Refresh();
            lblResult.Text = "Результат: ...";
        }


        private void InitializeModel()
        {

            List<Layer> loadedDenseLayers = Model.ModelSaver.LoadJson(@"model_config2.json");


            nn = new Model.Model();


            nn.Add(loadedDenseLayers[0]);
            nn.Add(new Model.Layers.ActivationReLU());

            nn.Add(loadedDenseLayers[1]);
            nn.Add(new Model.Layers.ActivationReLU());

            nn.Add(loadedDenseLayers[2]);
            nn.Add(new Model.Layers.ActivationSoftmax());

        }
    }
}

