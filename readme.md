## Leveraging Depth-Aware Representations for Few-Shot Traversability Estimation

> **Author**: Nils Schacknat  
> **Project page**: [nils-schacknat.github.io/thesis-demo](https://nils-schacknat.github.io/thesis-demo/)

This repository provides the code for my master's thesis, which addresses the task of vision-based traversability estimation.

 My thesis identifies Depth Anything v2 as an ideal task-aligned foundational model and proposes to leverage its depth-aware and semantically rich patch embeddings for few-shot segmentation. Further, it demonstrates that, with the Depth Anything v2 encoder as a backbone, a simple linear layer trained on only a few manual annotations can reliably predict traversability. A complementary instance selection strategy ensures that the training set remains both diverse and representative.

This method — termed DANTE (Depth ANything based Traversability Estimation) — requires minimal supervision and confidently surpasses recent pose-projection-based methods while running at real-time speeds.

### Setup
1. **Install the CUDA Toolkit**  
   Download and install the appropriate version from the [CUDA Toolkit website](https://developer.nvidia.com/cuda-downloads).

2. **Clone this repo and create a new Conda environment**
   ```bash
   git clone https://github.com/nils-schacknat/dante.git && cd dante
   conda create -n dante python=3.13
   conda activate dante

3. **Install PyTorch**  
   Visit the [PyTorch installation page](https://pytorch.org/get-started/locally/) and choose the correct version for your CUDA setup.

4. **Install the remaining dependencies**
   ```bash
   pip install -r requirements.txt 
   ```
&nbsp;
### Run the code

1. **Export the ViT backbone with TRT**  
   First, the Depth Anything v2 ViT backbone needs to be compiled with TensorRT to enable real-time inference.
   ```bash
   python export_model.py
   ```
   This should export `.trt` engines to a new `compiled_models/` directory.
   
2. **Train the linear decoder**  
   Second, a linear decoder needs to be trained on a handful of annotated images. This repository provides some examples under `assets/training_data`.
   ```bash
   python train.py
   ```
   This writes the trained weights to a file named `trained_decoder.pth`.

3. **Run inference**  
   This repo also provides a short validation video to run inference on.
   ```bash
   python inference.py
   ```
   This creates a new video `video_processed.mp4` with the overlayed predicted masks.

4. **Select representative frames**   
   Optionally, if you have your own video and want to extract representative frames for annotation, you can run `representative_image_selection.py`. It extracts $k$ representative images by running $k$-means clustering on the cosine similarities of the class tokens and selects the images closest to centers. 
   
