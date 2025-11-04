import torch, platform
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("bf16_supported:", torch.cuda.is_bf16_supported())
print("TF32 matmul:", torch.backends.cuda.matmul.allow_tf32)
print("float32 matmul precision:", torch.get_float32_matmul_precision())
try:
    import torchvision, diffusers, accelerate
    print("torchvision:", torchvision.__version__)
    print("diffusers:", getattr(diffusers,'__version__', 'n/a'))
    print("accelerate:", accelerate.__version__)
except Exception as e:
    print("import versions err:", e)
try:
    import xformers, xformers.ops
    print("xformers:", xformers.__version__, "mem_efficient_attn:", hasattr(xformers.ops,"memory_efficient_attention"))
except Exception as e:
    print("xformers: not available ->", e)
import os
print("PYTORCH_ALLOC_CONF:", os.environ.get("PYTORCH_ALLOC_CONF"))

from accelerate import Accelerator
acc = Accelerator()
print("world_size:", acc.num_processes, "rank:", acc.process_index, "mp:", acc.mixed_precision)
