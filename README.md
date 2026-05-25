# Hybrid OCR & Layout Analysis System (Vionet)

***This program is only for educational purposes!***

A high-performance desktop software complex designed for intelligent document layout analysis (Layout Analysis) and character-by-character optical character recognition (OCR) powered by a custom neural network engine built from scratch in C#.

The application provides a comprehensive **No-Code environment**, allowing users to visually design neural network topologies, manage dataset augmentation parameters, and execute asynchronous model training sessions directly within the graphical user interface.

---
## Lets see how basically it works

To train a neural network, different versions of gradient descent (GD) are used, and they are called optimizers. Their main essence is to manipulate gradients (a set of derivatives) to further reduce losses. Losses serve as our indicator of how much the model is working incorrectly. They have different natures, but the described Categorical Cross-Entropy Loss has an exponential complexity, so the worse the model, the greater the loss value. That is, our field becomes losses, and we need to reduce them to a number close to zero (it is worth noting that zero is also bad for the model, because then this is most likely overtraining). We can see vissually how it works on 3d graph, where our goal is to come down as much as possible:

<img width="600" height="400" alt="зображення" src="https://github.com/user-attachments/assets/344507c3-fb65-4afc-a171-dec2143340cc" />

In the case of the code, two main methods are implemented - Forward and Backward.
Forward describes the simple process of the neural network - the output is calculated using initialized weights and biases, and nonlinearity is added in the form of activation functions for network flexibility. As a result, we get our final answers. If the model is built on the principle of the last Softmax activation function - then the result will be the percentages of a particular class. That is, the result with the highest percentage is the one that the model considers the most correct, the prediction.

Backward describes a complex system of finding gradients based on principles of mathematical analysis (chain rule) and further improving the model based on this data. The basis is finding partial derivatives of weights, bias and inputs to find gradients across all neurons.

<img width="800" height="500" alt="зображення" src="https://github.com/user-attachments/assets/905049dc-5c7a-4d2f-8b92-8389049f0a40" />

The input layer in our case consists of 28x28 images, that is, a row of 784 pixels. The hidden layer is our hidden layers (dense) and activation functions. By going through them, the neural network selects the most influential pixels and gives them a greater weight (Weights), which affects the final result. Activation functions allow us to get out of the linear nature of neural networks (since calculations are performed in the form of Y = X * W + b), which provides incredible adaptation to complex structures. At the end – dense with a softmax activation function, which converts the sum of all neuron outputs to 1 (i.e. 100%), which allows us to concisely conduct tests and also easily conduct training.

We can see all classes in our diagram:

<img width="3343" height="1444" alt="ModelDiagram" src="https://github.com/user-attachments/assets/3c4af2ac-813e-47bf-acad-d293bfbaf308" />

---
## DATASETS

The __incredibly__ important part is the data. This project implements two methods for datasets: one of the most popular datasets for Latin and numbers – EMNIST; a self-created dataset for printed characters (the programmer can choose the language and symbols). The program code describes auxiliary DEBUG menus that allow you to easily check the correctness of the data, its appearance and possible errors.

The photo illustrates several examples from the EMNIST dataset, loged using the debug menu:

<img width="900" height="318" alt="зображення" src="https://github.com/user-attachments/assets/e133365d-23d9-4ab7-b781-25b05130d654" />

To simulate real-world document imperfections (such as paper texture, shaky hand scanning, or low-light noise), the engine executes live stochastic augmentation. The augmentation parameters are finely tuned via a modular configuration class:

```csharp
public class AugmentationConfig
{
    public bool UseBlur { get; set; } = true;
    public double BlurChance { get; set; } = 0.2;         // 20% probability to apply Gaussian blur
    public bool UseSaltPepper { get; set; } = true;
    public double SaltPepperIntensity { get; set; } = 0.01; // Percentage of pixel corruption
    public bool UseBrightness { get; set; } = true;
    public int BrightnessRange { get; set; } = 20;        // Shifts pixel brightness values
    public bool UseShift { get; set; } = true;
    public int MaxShift { get; set; } = 3;                // Random translation along X/Y axes (in pixels)
}
```

And here is the result after basic augmentation:

<img width="895" height="316" alt="зображення" src="https://github.com/user-attachments/assets/ef8f26a3-6751-411c-980c-419c1d53a34c" />

---
## Training

The train process is described in Trainer project. It's simple system of classes, where __Model__ include composition principe of OOP. 

> Augment data if needed -> Divide data into train/validate -> Choose Layers and their params -> Take one of the optim (and loss as well) -> Train and validate -> save model if you wish

In the end, you'll have all info about weaak classes (you can choose any treshold):

<img width="1000" height="531" alt="Screenshot_1" src="https://github.com/user-attachments/assets/cf603a1b-f039-46ae-912d-49e6ce53b949" />

---

## 🚀 Key Features & Capabilities

* **Hybrid Pipeline Architecture:** Combines classic geometric heuristics for ultra-fast document layout segmentation with an artificial neural network for linear text OCR.
* **Custom AI Engine (Vionet):** Full native implementation of dense layers (Fully Connected), activation functions (ReLU, Softmax), and backpropagation training mechanics with zero external Python/C++ library dependencies.
* **Intelligent Layout Analyzer:** Automatically separates document topography into three distinct semantic regions: `Text`, `MathFormula`, and `Image`.
* **No-Code Model Designer:** Comprehensive GUI for stacking neural network layers, configuring state-of-the-art optimizers (Adam, RMSProp, SGD), and tuning regularization parameters (Dropout).
* **Asynchronous Execution Model:** Multi-threaded processing utilizing `Task.Run` wraps all heavy mathematical workflows (training, augmentation, OCR pipelines) to ensure a fluid, non-blocking WPF UI thread experience.
---

## 🛠️ Technological Stack

* **Platform:** .NET 8.0 / .NET 9.0
* **Presentation Layer:** WPF (Windows Presentation Foundation)
* **Programming Language:** C#
* **AI & Mathematics Core:** Native `Vionet` framework (built from scratch, free of TensorFlow, PyTorch, or OpenCV wrappers).
* **Data Persistence:** System.Text.Json (manages deep export/import of topological layouts, weight/bias matrices, and character alphabet metadata).

---

## 📐 Architecture & Pipeline Breakdown

The core OCR processing pipeline operates across three granular abstraction levels:

### 1. Macro-Segmentation (Layout Topography Analysis)
* **Connected Components Labeling:** Segregates the binarized document bitmap into discrete continuous pixel clusters.
* **Spatial Ray-Casting & Heuristics:** Pinpoints mathematical formulas by detecting high vertical variance in symbol centroids, isolating fraction bars, and deploying local pattern matching to identify symbols like the equals sign (`=`).
* **Morphological Dilatation Clustering:** Uses an iterative `MergeAdjacentRegions` algorithm that inflates bounding boxes by a custom pixel buffer to group isolated words into unified, monolithic paragraph blocks.
* **Semantic Noise Masking:** Implements `GetMaskedTextBitmap` to mask out images and complex mathematical equations with solid white pixels before text OCR, neutralizing unexpected artifacts and boosting character accuracy.

### 2. Meso-Segmentation (Line Analysis)
* **Adaptive Binarization:** Standardizes resolution scales and dynamic contrasts.
* **Projection Profiling:** Cuts individual text strings out of a layout block using horizontal pixel density histograms.

### 3. Micro-Segmentation (Character OCR)
* **Glyph Assembly:** Discovers distinct characters and reconstructs multi-part symbols (such as specific regional glyphs or dot-accents) based on vertical coordinate bounds.
* **Tensor Normalization:** Centers and reshapes character bitmaps into structured $28 \times 28$ matrices, mapping pixel intensities smoothly from `0.0` to `1.0`.
* **Forward Propagation Inferencing:** Passes tensors through the neural network layers, where the final Softmax activation evaluates a probability distribution mapping to the corresponding alphabetical label character.

---

## 📂 Codebase Structure (`MainWindow` Partial Classes)

To enforce clean architectural boundaries and prevent single-file bloat, the `MainWindow` graphical logic is cleanly decoupled into dedicated structural files:
* `MainWindow.xaml.cs` — Application bootstrapping, core menus, and global lifecycle hooks.
* `MainWindow.Canvas.cs` — High-performance vector drawing overlays; manages interactive visual debugging via alpha-blended bounding boxes (`GetLayoutDebugBitmap`).
* `MainWindow.ImageOCR.cs` — The central traffic controller of the OCR pipeline, mediating data between the geometric layout analyzer and the AI engine.
* `MainWindow.Models.cs` — Handles state machines for loading, hot-swapping, and validating runtime configurations of independent neural networks.
* `MainWindow.Training.cs` — Drives asynchronous background training threads, tensor shape compatibility checks, and live dataset augmentation.

---

## ⚙️ Build and Installation Instructions

### Prerequisites
* Visual Studio 2022 (or JetBrains Rider)
* .NET SDK 8.0 / 9.0 or higher

### Steps to Run:
1. Clone the repository to your local workstation.
2. Open the solution file `.sln` in your preferred IDE.
3. Ensure your model weights are mapped correctly within the `NeuralNetworks/` subfolder.
4. Switch your build configuration to **Release** to enable compiler optimizations for maximum matrix processing speeds.
5. Press `F5` to build and run the application.

---
