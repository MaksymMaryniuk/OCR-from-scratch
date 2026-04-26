using Model.Layers;
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;

namespace Model
{
    public static class ModelSaver
    {
        public static void SaveJson(string filePath, List<Layer> layers)
        {
            var config = new ModelConfig();

            foreach (var layer in layers)
            {
                if (layer is Layer_Dense dense)
                {
                    int rows = dense.Weights.GetLength(0);
                    int cols = dense.Weights.GetLength(1);

                    float[][] weightExport = new float[rows][];
                    for (int i = 0; i < rows; i++)
                    {
                        weightExport[i] = new float[cols];
                        for (int j = 0; j < cols; j++)
                            weightExport[i][j] = dense.Weights[i, j];
                    }

                    config.Layers.Add(new LayerData
                    {
                        Rows = rows,
                        Cols = cols,
                        Weights = weightExport,
                        Biases = dense.Biases
                    });
                }
            }

            var options = new JsonSerializerOptions { WriteIndented = true };
            string jsonString = JsonSerializer.Serialize(config, options);
            File.WriteAllText(filePath, jsonString);

            Console.WriteLine($"Saved in JSON: {filePath}");
        }

        public static List<Layer> LoadJson(string filePath)
        {
            string jsonString = File.ReadAllText(filePath);
            var config = JsonSerializer.Deserialize<ModelConfig>(jsonString);
            var layers = new List<Layer>();

            foreach (var layerData in config.Layers)
            {
                if (layerData.Type == "DENSE")
                {
                    var layer = new Layer_Dense(layerData.Rows, layerData.Cols);

                    for (int i = 0; i < layerData.Rows; i++)
                    {
                        for (int j = 0; j < layerData.Cols; j++)
                        {
                            layer.Weights[i, j] = layerData.Weights[i][j];
                        }
                    }

                    Array.Copy(layerData.Biases, layer.Biases, layerData.Cols);
                    layers.Add(layer);
                }
            }

            return layers;
        }


        private class LayerData
        {
            public string Type { get; set; } = "DENSE";
            public int Rows { get; set; }
            public int Cols { get; set; }
            public float[][] Weights { get; set; }
            public float[] Biases { get; set; }
        }

        private class ModelConfig
        {
            public string ModelName { get; set; } = "EMNIST_Model";
            public string CreatedAt { get; set; } = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            public List<LayerData> Layers { get; set; } = new List<LayerData>();
        }

    }
}
