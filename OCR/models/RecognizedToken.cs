using System.Windows.Media;

namespace OCR.models
{
    public class RecognizedToken
    {
        public char   Symbol     { get; set; }
        public float  Confidence { get; set; }
        public SolidColorBrush ConfidenceBrush
        {
            get
            {
                byte red   = (byte)(255 * (1f - Confidence));
                byte green = (byte)(255 * Confidence);
                return new SolidColorBrush(Color.FromRgb(red, green, 0));
            }
        }
    }
}
