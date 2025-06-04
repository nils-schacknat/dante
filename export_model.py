import torchvision.transforms.v2 as transforms
import time
import os
from pathlib import Path

import torch
import tensorrt as trt

class DinoV2(torch.nn.Module):
    def __init__(self, model_type, reg=True):
        super().__init__()
        reg = "_reg" if reg else ""
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_type}14{reg}').to("cpu").eval()
        self.dinov2.interpolate_antialias = False  # TensorRT does not compile the model with antialias enabled
        self.normalization_transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.name = f"dinov2_{model_type}{reg}"

    def forward(self, x):
        with torch.no_grad():
            ((intermediate_features, cls_token),) = self.dinov2.get_intermediate_layers(
                x, 
                reshape=True,
                return_class_token=True
            )
        return intermediate_features, cls_token
    
class DepthAnythingV2(DinoV2):
    def __init__(self, model_type):
        super().__init__(model_type, reg=False)

        checkpoint_path = Path(f"depth_anything_v2_{model_type}.pth")
        if not checkpoint_path.exists():
            model_name = dict(s="Small", b="Base", l="Large")[model_type]
            url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model_name}/resolve/main/depth_anything_v2_vit{model_type}.pth?download=true"
            os.system(f"wget -O {checkpoint_path} \"{url}\"")

        # Load the depth anything v2 checkpoint into the dinov2-vit model
        checkpoint = torch.load(checkpoint_path)
        prefix = "pretrained."
        state_dict = {
            k[len(prefix):]: v for k, v in checkpoint.items()
            if k.startswith(prefix)
        }
        self.dinov2.load_state_dict(state_dict)
        self.name = f"depth_anything_v2_{model_type}"

def export(
    save_path: str,
    input_size: int,
    model
):
    """
    save_path: Directory to save the model
    input_size: Width and height of the input image
    model: The model to compile with TensorRT
    """

    os.makedirs(save_path, exist_ok=True)

    # Create dummy input (use batch_size=1 for tracing, even if dynamic)
    dummy_input = torch.ones((1, 3, *input_size))

    # Warmup
    _ = model(dummy_input)

    onnx_path = Path(save_path) / f"{model.name}_{input_size[0]}x{input_size[1]}.onnx"

    output_names = ["features", "cls_token"]
    dynamic_axes = {
        "input": {0: "batch_size"},
        "features": {0: "batch_size"},
        "cls_token": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"Model exported to {onnx_path}")
    time.sleep(2)

    # ONNX to TensorRT
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for error_idx in range(parser.num_errors):
                print(parser.get_error(error_idx))
            raise ValueError("Failed to parse the ONNX model.")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8 GB

    # Define optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()
    min_shape = (1, 3, *input_size)
    opt_shape = (1, 3, *input_size)
    max_shape = (16, 3, *input_size)

    profile.set_shape("input", min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    with open(onnx_path.with_suffix(".trt"), "wb") as f:
        f.write(serialized_engine)

if __name__ == '__main__':
    torch.cuda.set_device(2)
    export(
        save_path="compiled_models",
        input_size=(518, 518),
        model=DepthAnythingV2("s"),
    )
    export(
        save_path="compiled_models",
        input_size=(518, 924),
        model=DepthAnythingV2("s"),
    )
