import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import cv2
import torch
import torch.nn.functional as F


class TensorRTModel:
    """
    This class loads the compiled TRT engine.
    """
    def __init__(
        self,
        engine_path: str,
        input_shape=(1, 3, 518, 518),
        device=0,
        logger_level=trt.Logger.WARNING,
    ):
        """
        Args:
            engine_path (str): Path to the TRT engine file.
            input_shape (tuple): Shape of the input tensors. The batch size is dynamic,
                channels, height and width are fixed and need to be the same as the model has
                been compiled with.
            device (int): The gpu to use (if multiple are available).
            logger_level (int): The logger level.
        """
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.logger = trt.Logger(logger_level)
        self.device = torch.device(type="cuda", index=device)

        # Create a PyCUDA context on `device`
        cuda.init()  # Initialize the CUDA driver (if not already done)
        self.cuda_device = cuda.Device(device)
        self.ctx = self.cuda_device.make_context()  # push a new context on GPU `device`
        _ = torch.zeros(0, device=self.device)  # force CUDA context init

        # Load serialized engine into TRT runtime
        self.runtime = trt.Runtime(self.logger)
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate GPU buffers (torch tensors) for all I/O bindings
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def __del__(self):
        # Pop & destroy the PyCUDA context when the object goes out of scope
        try:
            self.ctx.pop()  # Pop the context that was pushed in __init__
        except Exception:
            pass  # In some interpreter-shutdown scenarios, cuda may already be torn down.

    def _allocate_buffers(self):
        """
        For each binding (input or output), we create a torch.cuda.Tensor of the exact shape
        needed (with dtype matching TRT’s expectation), record its device pointer, and store it.
        """
        inputs = []
        outputs = []
        n_bindings = self.engine.num_io_tensors
        bindings = [None] * n_bindings
        stream = cuda.Stream()

        for idx in range(n_bindings):
            name = self.engine.get_tensor_name(idx)
            trt_dtype = self.engine.get_tensor_dtype(name)
            np_dtype = trt.nptype(trt_dtype)

            # If the binding has a dynamic dimension (-1), we fix it now to self.input_shape.
            shape = self.context.get_tensor_shape(name)
            if -1 in shape:
                # Only allow dynamic batch along dim0; we fix shape to input_shape
                self.context.set_input_shape(name, self.input_shape)
                shape = self.context.get_tensor_shape(name)

            # Compute total element count, e.g. N*C*H*W
            elem_count = trt.volume(shape)  # this is an int
            # Construct a torch tensor on GPU with the same dtype & shape
            # Note: trt.nptype(...) gives something like numpy.float32 → we map to torch.float32
            torch_dtype = {
                np.dtype("float32"): torch.float32,
                np.dtype("int32"): torch.int32,
                np.dtype("float16"): torch.float16,
            }[np.dtype(np_dtype)]

            # Create a zeroed tensor on the correct CUDA device:
            t = torch.zeros(tuple(shape), dtype=torch_dtype, device=self.device)
            # Record its device pointer
            device_ptr = t.data_ptr()

            # The binding expects a raw pointer (int) to GPU memory:
            bindings[idx] = int(device_ptr)

            # Figure out if this is an input or output:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({"name": name, "tensor": t})
            else:
                outputs.append({"name": name, "tensor": t})

        return inputs, outputs, bindings, stream

    def infer(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on the given input tensor. The tensor should be correctly transformed
        through self.transform.

        Args:
            input_tensor (torch.Tensor): The input tensor to be processed.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The output tensors.
        """
        # Cast to torch.float32 if necessary
        if input_tensor.dtype != torch.float32:
            input_tensor = input_tensor.to(torch.float32)

        # Check device
        assert input_tensor.device == self.device, f"Expected input tensor device is {self.device}, got {input_tensor.device}"
        batch_size = input_tensor.shape[0]

        # If the provided tensor's batch size is smaller than the engine's batch size,
        # it is zero-padded along the batch dimension
        if batch_size < self.input_shape[0]:
            pad_batch = self.input_shape[0] - batch_size
            padding = torch.zeros(
                (pad_batch, *input_tensor.shape[1:]),
                dtype=input_tensor.dtype,
                device=self.device,
            )
            input_tensor = torch.cat([input_tensor, padding], dim=0)

        elif batch_size > self.input_shape[0]:
            raise ValueError(
                f"Input batch size {batch_size} > expected {self.input_shape[0]}"
            )

        # Make sure it’s contiguous (so data_ptr() matches layout TRT expects).
        input_tensor = input_tensor.contiguous()

        # Bind the input tensor’s device pointer:
        self.context.set_tensor_address(
            self.inputs[0]["name"], int(input_tensor.data_ptr())
        )

        # Bind each preallocated output tensor’s device pointer:
        for out_dict in self.outputs:
            name = out_dict["name"]
            t = out_dict["tensor"]
            self.context.set_tensor_address(name, int(t.data_ptr()))

        # Execute inference asynchronously
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        # Wait for TRT to finish writing into our output tensors
        self.stream.synchronize()

        # At this point, self.outputs[i]["tensor"] already holds the output on CUDA.
        # We can slice off any padded rows if batch < engine_batch
        results = []
        for out_dict in self.outputs:
            full_tensor = out_dict["tensor"]
            results.append(full_tensor[:batch_size])

        return results

    def benchmark(self, num_runs=100):
        """
        Runs inference for the given number and reports the average time.

        Args:
            num_runs (int): Number of times to run the inference.
        """
        # Create a random CUDA tensor of shape input_shape:
        input_cuda = torch.randn(self.input_shape, device=self.device, dtype=torch.float32)
        total_time = 0.0
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            self.infer(input_cuda)
            torch.cuda.synchronize()
            total_time += time.time() - start

        avg_time_ms = (total_time / num_runs) * 1000 / self.input_shape[0]
        fps = 1000.0 / avg_time_ms
        print(f"Average per-image inference time: {avg_time_ms:.2f} ms")
        print(f"FPS: {fps:.2f}")

    def transform_cpu(self, image: np.ndarray) -> np.ndarray:
        """
        On the cpu, transforms an input image to enable inference.

        Args:
            image (np.ndarray): The input image.
        """
        # Resize to specified height and width
        _, _, h, w = self.input_shape
        image = cv2.resize(image, (w, h))

        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_transformed = (image.astype(np.float32) / 255.0 - mean) / std

        # Transpose
        return image_transformed.transpose(2, 0, 1)

    def transform(self, image: np.ndarray) -> torch.Tensor:
        """
        On the gpu, transforms an input image to enable inference.

        Args:
            image (np.ndarray): The input image.
        """
        # Transform the image to a Tensor and move to the gpu
        img_t = torch.from_numpy(image).to(self.device)

        # Transpose
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)

        # Resize to specified height and width
        img_t = img_t.float() / 255.0
        _, _, h, w = self.input_shape
        img_resized = F.interpolate(img_t, size=(h, w), mode="bilinear", align_corners=False)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_resized.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=img_resized.device).view(1, 3, 1, 1)
        image_transformed = (img_resized - mean) / std
        return image_transformed


def load_backbone(
    input_size: tuple[int, int],
    batch_size=1,
    engine_type="depth_anything_v2_s",
    device=0
):
    """
    A wrapper which loads the TRT model.

    Args:
        input_size (tuple[int, int]): The height and width of the input tensors.
        batch_size (int): The batch size.
        engine_type (str): The name of the backbone.
        device (int): The gpu to use (if multiple are available).

    Returns:
        (TensorRTModel) The loaded model.
    """
    engine_path = f"compiled_models/{engine_type}_{input_size[0]}x{input_size[1]}.trt"
    return TensorRTModel(engine_path, input_shape=(batch_size, 3, *input_size), device=device)


if __name__ == "__main__":
    # The input dimensions the model backbone has been compiled with
    input_size = (518, 924)
    # The batch size to initialize the backbone with
    batch_size = 1

    engine_path = f"compiled_models/depth_anything_v2_s_{input_size[0]}x{input_size[1]}.trt"
    model = TensorRTModel(engine_path, input_shape=(batch_size, 3, *input_size))

    dummy_input = torch.rand(batch_size, 3, *input_size).to(model.device)
    features, cls_token = model.infer(dummy_input)
    print(f"features shape: {features.shape}, cls_token shape:{cls_token.shape}")
    model.benchmark()
