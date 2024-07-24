import torch

@torch.no_grad()
def generate_unstructured_sparse_tensor(t: torch.Tensor, sparsity: float) -> torch.Tensor:
    t = t.clone()
    nz_el = int(sparsity*t.numel())
    idx = torch.randperm(t.numel())
    t.view(-1)[idx[:nz_el]] = 0
    return t


@torch.no_grad()
def generate_structured_sparse_tensor(t: torch.Tensor, sparsity: float) -> torch.Tensor:
    t = t.clone()
    num_zero_rows = int(sparsity*len(t))
    idx = torch.randperm(len(t))
    t[idx[:num_zero_rows]] = 0
    return t
    
@torch.no_grad()
def generate_ffi_structure(t: torch.Tensor, sparsity: float) -> nn.Linear:
    t = t.clone()
    n_zeros = int(t.numel() * (sparsity))
    n_zeros_per_neuron = n_zeros // t.shape[0]
    for idx, neuron in enumerate(t):
        rand_idx = torch.randperm(n=len(neuron))
        t[idx, rand_idx[:n_zeros_per_neuron-1]] = 0
    return t

def assert_ffi(t: torch.Tensor):
    ffi = (t[0]!=0).sum()
    for n in t:
        assert (n!=0).sum()==ffi

def print_sparsity(t: torch.Tensor):
    return f"{(t==0).sum()/t.numel():.2f}"