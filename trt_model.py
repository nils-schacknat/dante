import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import cv2
import torch
import torch.nn.functional as F


class TensorRTModel:
    def __init__(self, engine_path, input_shape=(1, 3, 518, 518), device=0, logger_level=trt.Logger.WARNING):
        self.engine_path = engine_path
        self.input_shape = input_shape  # NCHW
        self.logger = trt.Logger(logger_level)

        torch.cuda.set_device(device)
        _ = torch.zeros(0, device=f"cuda")

        # Load engine and context
        self.runtime = trt.Runtime(self.logger)
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = [None] * self.engine.num_io_tensors
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            shape = self.context.get_tensor_shape(name)
            if -1 in shape:
                self.context.set_input_shape(name, self.input_shape)
                shape = self.context.get_tensor_shape(name)

            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings[i] = int(device_mem)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({'name': name, 'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'name': name, 'host': host_mem, 'device': device_mem})

        return inputs, outputs, bindings, stream

    def infer(self, input_tensor):
        batch_size = input_tensor.shape[0]
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.cpu().numpy()

        if batch_size < self.input_shape[0]:
            # Expand input tensor with zeros to match the expected batch size
            pad_shape = (self.input_shape[0] - input_tensor.shape[0],) + input_tensor.shape[1:]
            padding = np.zeros(pad_shape, dtype=input_tensor.dtype)
            input_tensor = np.concatenate([input_tensor, padding], axis=0)

        elif batch_size > self.input_shape[0]:
            # Raise error because batch size exceeds defined batch size
            raise ValueError(f"Input batch size {input_tensor.shape[0]} exceeds expected batch size {self.input_shape[0]}")

        # Flatten and copy input to host memory
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())

        # Copy input to device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        for tensor in self.inputs + self.outputs:
            self.context.set_tensor_address(tensor['name'], tensor['device'])

        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs back to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()

        # Parse the outputs
        results = []
        for out in self.outputs:
            name = out['name']
            shape = self.context.get_tensor_shape(name)
            array = np.array(out['host']).reshape(shape)[:batch_size]
            results.append(torch.from_numpy(array).cuda())
        return results
    
    def benchmark(self, num_runs=100):
        input_tensor = np.random.rand(*self.input_shape).astype(np.float32)
        total_time = 0.0
        for _ in range(num_runs):
            start = time.time()
            self.infer(input_tensor)
            total_time += time.time() - start
        avg_time_ms = (total_time / num_runs) * 1000 / self.input_shape[0]
        fps = 1000.0 / avg_time_ms
        print(f"Average Inference Time: {avg_time_ms:.2f} ms")
        print(f"FPS (Frames per Second): {fps:.2f}")

    def transform_(self, image):
        _, _, h, w = self.input_shape
        image = cv2.resize(image, (w, h))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_transformed = (image.astype(np.float32) / 255.0 - mean) / std
        return image_transformed.transpose(2, 0, 1)[None]
    
    def transform(self, image: np.ndarray) -> torch.Tensor:
        img_t = torch.from_numpy(image).to("cuda")
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)
        img_t = img_t.float() / 255.0
        _, _, h, w = self.input_shape
        img_resized = F.interpolate(img_t, size=(h, w), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_resized.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=img_resized.device).view(1, 3, 1, 1)
        image_transformed = (img_resized - mean) / std
        return image_transformed
    

def load_backbone(
    input_size,
    batch_size=1,
    engine_type="depth_anything_v2_s",
    device=0
):
    engine_path = f"compiled_models/{engine_type}_{input_size[0]}x{input_size[1]}.trt"
    return TensorRTModel(engine_path, input_shape=(batch_size, 3, *input_size), device=device)


# Example usage
if __name__ == "__main__":
    batch_size = 1
    input_size = (518, 518)
    engine_path = f"compiled_models/depth_anything_v2_s_{input_size[0]}x{input_size[1]}.trt"
    model = TensorRTModel(engine_path, input_shape=(batch_size, 3, *input_size))

    dummy_input = np.random.rand(batch_size, 3, *input_size).astype(np.float32)
    features, cls_token = model.infer(dummy_input)
    print(f"features shape: {features.shape}, cls_token shape:{cls_token.shape}")
    model.benchmark()
