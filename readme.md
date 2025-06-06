## Leveraging Depth-Aware Representations for Few-Shot Traversability Estimation

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

### Run the code

1. **Export the ViT backbone with TRT**  
   First, the Depth Anything v2 ViT backbone needs to be compiled with TensorRT to enable real-time inference.
   ```bash
   python export_model.py
   ```
   This should export `.trt` engines under a new `compiled_models/` directory.
   
2. **Train the linear decoder**  
   Second, the linear decoder needs to be trained on a handful of annotated images. This repository provides some examples under `assets/training_data`.
   ```bash
   python train.py
   ```
   This should write the trained weights to a file named `trained_decoder.pth`.

3. **Run inference**
   This repo also provides a short testing video to run inference one.
   ```bash
   python inference.py
   ```
   This creates a new video with the overlayed predicted masks at `video_processed.mp4`

4. **Select representative frames**   
   Optionally, if you have your own video and automatically want to extract representative frames for annotation, you can take a look at `representative_image_selection.py`. It extracts $k$ representative images by running $k$-means clustering on the cosine similarities of the class tokens and selects the images closest to centers. 
   
