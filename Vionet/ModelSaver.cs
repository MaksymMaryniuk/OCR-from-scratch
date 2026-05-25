using Vionet.Layers;
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;

namespace Vionet
{
    public static class ModelSaver
    {
        public static void SaveJson(string filePath, List<Layer> layers, string name = "No-Name", string labels = null)
        {
            var config = new ModelConfig
            {
                ModelName = name,
                Labels = labels
            };

            foreach (var layer in layers)
            {
                if (!layer.IsTrainable)
                    continue;

                var data = new LayerData
                {
                    Type = layer.Type
                };

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

                    data.Rows = rows;
                    data.Cols = cols;
                    data.Weights = weightExport;
                    data.Biases = dense.Biases;
                }

                config.Layers.Add(data);
            }

            var options = new JsonSerializerOptions { WriteIndented = true };
            string jsonString = JsonSerializer.Serialize(config, options);
            File.WriteAllText(filePath, jsonString);

            Console.WriteLine($"Saved in JSON: {filePath}");
        }

        public static Model LoadJson(string filePath)
        {
            string jsonString = File.ReadAllText(filePath);
            var config = JsonSerializer.Deserialize<ModelConfig>(jsonString);

            var model = new Model();

            if (config.Labels != null)
            {
                model.Labels = config.Labels;
            }

            foreach (var layerData in config.Layers)
            {
                Layer layer = layerData.Type switch
                {
                    "DENSE" => CreateDense(layerData),
                    "RELU" => new ActivationReLU(),
                    "SOFTMAX" => new ActivationSoftmax(),
                    "DROPOUT" => new Layer_Dropout(0f),
                    _ => throw new Exception($"Unknown layer type: {layerData.Type}")
                };

                model.Add(layer);
            }

            return model;
        }


        private class LayerData
        {
            public string Type { get; set; }

            public int Rows { get; set; }
            public int Cols { get; set; }

            public float[][] Weights { get; set; }
            public float[] Biases { get; set; }
        }

        private class ModelConfig
        {
            public string ModelName { get; set; } = "No-Name";
            public string Labels { get; set; }
            public string CreatedAt { get; set; } = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            public List<LayerData> Layers { get; set; } = new List<LayerData>();
        }


        private static Layer_Dense CreateDense(LayerData data)
        {
            var layer = new Layer_Dense(data.Rows, data.Cols);

            for (int i = 0; i < data.Rows; i++)
                for (int j = 0; j < data.Cols; j++)
                    layer.Weights[i, j] = data.Weights[i][j];

            Array.Copy(data.Biases, layer.Biases, data.Cols);

            return layer;
        }
    }
}
