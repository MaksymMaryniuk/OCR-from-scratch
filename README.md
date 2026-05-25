# Hybrid OCR & Layout Analysis System (Vionet)

***This program is only for educational purposes!***

A high-performance desktop software complex designed for intelligent document layout analysis (Layout Analysis) and character-by-character optical character recognition (OCR) powered by a custom neural network engine built from scratch in C#.

The application provides a comprehensive **No-Code environment**, allowing users to visually design neural network topologies, manage dataset augmentation parameters, and execute asynchronous model training sessions directly within the graphical user interface.

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
