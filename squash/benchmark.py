import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from torch import _dynamo as dynamo
from torch import _C
from functools import reduce
from copy import deepcopy
from transformers import OPTForCausalLM, GPT2Tokenizer, OPTModel
import pathlib
import pickle

from squash.ffi_linear import FFILinear
from squash.utils import ffi_magnitude_pruner

def patch_model(model: nn.Module, dense_to_sparse_dict: dict[str, nn.Module]) -> None:
    for dense_name, sparse_mod in dense_to_sparse_dict.items():
        setattr(reduce(getattr, dense_name.split(".")[:-1], model), dense_name.split(".")[-1], sparse_mod)

@torch.no_grad()
def benchmark_causal_model(model, model_inputs, sub_label: str, description: str):
    for _ in range(2):
        _  = model.generate(**model_inputs, max_new_tokens=1, do_sample=False)
    result = benchmark.Timer(
        stmt="model.generate(**model_inputs, max_new_tokens=1, do_sample=False)",
        setup="",
        globals={"model": model, "model_inputs": model_inputs},
        label="OPT-350M Test",
        sub_label=sub_label,
        description=description,
        num_threads=4,
    ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
    return result


@torch.no_grad()
def benchmark_model(model, model_inputs, sub_label: str, description: str):
    for _ in range(2):
        _  = model(**model_inputs)
    result = benchmark.Timer(
        stmt="model(**model_inputs)",
        setup="",
        globals={"model": model, "model_inputs": model_inputs},
        label="OPT-350M Test",
        sub_label=sub_label,
        description=description,
        num_threads=4,
    ).blocked_autorange(min_run_time=__MIN_RUN_TIME)
    return result


def get_sparse_model(dense_model, sparsity, dtype):
    sparse_model = deepcopy(dense_model)
    sparse_model_pickle_fname = pathlib.Path.cwd() / "sparse_opt.pkl"
    if sparse_model_pickle_fname.is_file():
        with open(sparse_model_pickle_fname, mode="rb") as handle:
            return pickle.load(handle)
    sparse_model = deepcopy(dense_model)
    dense_sparse_mapping = {}
    for n,m in sparse_model.named_modules():
        if isinstance(m, nn.Linear):
            print(f"Sparsifying module: {n}")
            m.weight.data = ffi_magnitude_pruner(m.weight, sparsity)
            dense_sparse_mapping[n] = FFILinear(m, dtype=dtype)
    patch_model(sparse_model, dense_sparse_mapping)
    with open(sparse_model_pickle_fname, "wb") as handle:
        pickle.dump(sparse_model, handle)
    return sparse_model


if __name__ == "__main__":
    dynamo.config.verbose = True
    compiler_kwargs = {
        "mode": "max-autotune",
        # "mode": "reduce-overhead",
        # "fullgraph": True,
    }
    torch.jit.enable_onednn_fusion(
        True
    )
    torch.set_float32_matmul_precision('high')
    __MIN_RUN_TIME = 2
    _DTYPE = torch.float32
    device = torch.device("cuda")
    sparsities = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    # batch_sizes = [2**n for n in range(1, 7)]
    batch_sizes = [2**n for n in range(1, 2)]
    sequence_lens = [512]
    
    sparsity = 0.90
    batch_size = 32
    seq_len = 512
    results = []
    # model = OPTForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=_DTYPE)
    model = OPTModel.from_pretrained("facebook/opt-350m", torch_dtype=_DTYPE)
    sparse_model = get_sparse_model(model, sparsity, _DTYPE)
    tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.eval().to(device)
    sparse_model.eval().to(device)
    compiled_dense = torch.compile(model, backend="inductor", **compiler_kwargs)  
    compiled_sparse = torch.compile(sparse_model, backend="inductor", **compiler_kwargs)  

    for batch_size in batch_sizes:
        print(f"Benchmarking batch size == {batch_size}")
        prompt = ["A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the "   
            "Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived "
            "there?" for _ in range(batch_size)
        ]
        model_inputs = tokenizer(prompt, padding=True, return_tensors="pt", pad_to_multiple_of = seq_len).to(device)
        results.append(benchmark_model(model, model_inputs, f"Batch_size={batch_size}", "dense"))
        results.append(benchmark_model(compiled_dense, model_inputs, f"Batch_size={batch_size}", "dense_compiled"))
        results.append(benchmark_model(sparse_model, model_inputs, f"Batch_size={batch_size}", "sparse"))
        results.append(benchmark_model(compiled_sparse, model_inputs, f"Batch_size={batch_size}", "sparse_compiled"))
    compare = benchmark.Compare(results)
    compare.colorize()
    print(compare)