#optional Triton demo
try:
    import triton
    import triton.language as tl
    import torch
except Exception:
    triton = None
    tl = None  # keep name defined


# A tiny Triton ReLU kernel (conceptual)
if triton:

    @triton.jit
    def relu_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X_ptr + offs, mask=mask)
        y = tl.maximum(x, 0)
        tl.store(Y_ptr + offs, y, mask=mask)

    def triton_relu(x: torch.Tensor) -> torch.Tensor:
        """Elementwise ReLU using a Triton kernel. Expects CUDA tensor."""
        if triton is None:
            raise RuntimeError("Triton not installed")
        if not x.is_cuda:
            raise RuntimeError("Input must be a CUDA tensor")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("dtype must be float16/bfloat16/float32")

        x = x.contiguous()
        y = torch.empty_like(x)

        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)

        relu_kernel[grid](x, y, N, BLOCK=BLOCK)
        return y
