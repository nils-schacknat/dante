import torchvision.transforms.v2 as transforms
import time
import os
from pathlib import Path
import torch
import tensorrt as trt


class DinoV2(torch.nn.Module):
    """
    A wrapper for the DINOv2 model.
    """
    def __init__(self, model_type: str, reg=True):
        """
        Args:
            model_type (str): The ViT type to use (s, b, l).
            reg (bool): Whether the model should use register tokens.
        """
        super().__init__()
        reg = "_reg" if reg else ""
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_type}14{reg}').to("cpu").eval()

        # Disable this because TensorRT does not compile the model with antialias enabled
        self.dinov2.interpolate_antialias = False
        self.name = f"dinov2_{model_type}{reg}"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs inference with the DINOv2 model and returns the spatial features of the last layer
        and the class token.

        Args:
            x (torch.Tensor): The input image, already transformed. Height and width need
                to be multiples of 14.

        Returns:
            tuple[torch.Tensor, torch.Tensor] The spatial features and the class token of the last layer.
        """
        with torch.no_grad():
            ((features, cls_token),) = self.dinov2.get_intermediate_layers(
                x, 
                reshape=True,
                return_class_token=True
            )
        return features, cls_token


class DepthAnythingV2(DinoV2):
    """
    A wrapper for the DepthAnythingV2 model that serves as the backbone for DANTE.
    """
    def __init__(self, model_type: str):
        """
        Args:
            model_type (str): The ViT type to use (s, b, l).
        """
        super().__init__(model_type, reg=False)

        checkpoint_path = Path(f"depth_anything_v2_{model_type}.pth")
        # If the checkpoint does not exist, then download it
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


def export(save_path: str, input_size: tuple[int, int], model: DinoV2):
    """
    Exports the model to onnx and compiles it with TensorRT.

    Args:
        save_path (str): The directory to save the compiled model to.
        input_size (tuple[int, int]): Height and width of the input tensors.
            Need to be multiples of 14.
        model: The model to export and compile.
    """

    os.makedirs(save_path, exist_ok=True)

    # Create dummy input (use batch_size=1 for tracing, even if dynamic)
    dummy_input = torch.ones((1, 3, *input_size))
    # Warmup
    _ = model(dummy_input)

    # Set the onnx path, the names of the model outputs and the dynamic axes (batch dimension)
    onnx_path = Path(save_path) / f"{model.name}_{input_size[0]}x{input_size[1]}.onnx"
    output_names = ["features", "cls_token"]
    dynamic_axes = {
        "input": {0: "batch_size"},
        "features": {0: "batch_size"},
        "cls_token": {0: "batch_size"},
    }

    # Export the model with onnx
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
    # Compile with single precision (FP16)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)

    # Define optimization profile for dynamic batch size, optimal batch size is set to one.
    # The maximum allowed batch size is 16.
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
    # The gpu the model should be exported on (if multiple are available)
    torch.cuda.set_device(0)

    # Export the Depth Anything v2 Small backbone. Square resolution for training and
    # rectangular resolution for inference (924/518 ≈ 1280/720).
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
