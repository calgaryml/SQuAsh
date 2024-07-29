import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from torch import _dynamo as dynamo
from functools import reduce

from transformers import OPTForCausalLM, GPT2Tokenizer

from squash.ffi_linear import FFILinear
from squash.utils import ffi_magnitude_pruner

def patch_model(model: nn.Module, dense_to_sparse_dict: dict[str, nn.Module]) -> None:
    for dense_name, sparse_mod in dense_to_sparse_dict.items():
        setattr(reduce(getattr, dense_name.split(".")[:-1], model), dense_name.split(".")[-1], sparse_mod)

def benchmark_model(model, model_inputs, sub_label: str, description: str):
    benchmark.Timer(
        stmt="model.generate(**model_inputs, max_new_tokens=30, do_sample=False)",
        setup="",
        globals={"model": model, "model_inputs": model_inputs},
        label="OPT-350M Test",
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
    
    

if __name__ == "__main__":
    dynamo.config.verbose = True
    compiler_kwargs = {
        "mode": "max-autotune",
        "fullgraph": True,
    }
    torch.jit.enable_onednn_fusion(
        True
    )
    __MIN_RUN_TIME = 2
    _DTYPE = torch.float32
    device = torch.device("cuda")
    sparsities = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    model = OPTForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=_DTYPE)
    sparse_model = OPTForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=_DTYPE)

    dense_sparse_mapping = {}
    for n,m in sparse_model.named_modules():
        if isinstance(m, nn.Linear):
            print(f"Sparsifying module: {n}")
            m.weight.data = ffi_magnitude_pruner(m.weight, .9)
            dense_sparse_mapping[n] = FFILinear(m, dtype=_DTYPE)
    
    for dense, sparse in dense_sparse_mapping.items():
        patch_model(sparse_model, dense_sparse_mapping)
    
    
    tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

    prompt = ["A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the "   
            "Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived "
            "there?",
            "hello world!"]


    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    model_inputs
    # model.to(device)
    sparse_model.to(device)
    generated_ids = sparse_model.generate(**model_inputs, max_new_tokens=30, do_sample=False)
    print(tokenizer.batch_decode(generated_ids))
