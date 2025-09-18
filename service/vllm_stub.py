#optional route wiring
# Minimal vLLM placeholder to show CPU path availability
try:
    from vllm import LLM, SamplingParams
    VLLM_OK = True
except Exception:
    VLLM_OK = False


class TinyLLM:
    def __init__(self):
        if VLLM_OK:
            # NOTE: facebook/opt-125m is a relatively small model, but may still be heavy.
            # Consider lazy-loading or mocking for tests.
            self.llm = LLM(model="facebook/opt-125m")
        else:
            self.llm = None

    def generate(self, prompt: str) -> str:
        """Generate text using vLLM if available, otherwise return stub output."""
        if not self.llm:
            return "[stub] vLLM not installed; returning canned output"

        out = self.llm.generate([prompt], SamplingParams(max_tokens=32))
        return out[0].outputs[0].text
