"""Run ONCE (with the `nano` branch's mamba_ssm active) to generate the
shared random weights + input patterns that both branches will be tested
against. Saves to shared_data.pt next to this script.

is_mimo=False (SISO only), dtype=torch.float16 per the current test scope.
"""

import os
import torch
from mamba_ssm import Mamba3

SEED = 123
PATTERN, BATCH, LENGTH, DIM = 10, 2, 2048, 768
DTYPE = torch.float16

# The meaning of each parameter can be found in
# mamba3.py of the mamba3-note branch
CONFIG = dict(
    d_model=DIM,
    d_state=128,
    #expand=2,
    headdim=64,
    is_mimo=False,
    mimo_rank=4,
    chunk_size=16, # 64/mimo_rank if x is in bf16, else 32/mimo_rank
    is_outproj_norm=False,
    dtype=DTYPE,
    # comment this line to make ngroups to be default value (=1)
    #ngroups=32,
    #rope_fraction=0.5, # redundant: rope_fraction is 0.5 by default
)


def main():
    torch.manual_seed(SEED)
    # `device = "cuda"` tells pyTorch to put the random generated x directly on GPU
    x = torch.randn(PATTERN, BATCH, LENGTH, DIM, dtype=DTYPE, device="cuda")

    model = Mamba3(**CONFIG).to("cuda")

    # B_bias and C_bias default to a constant 1.0 (see mamba3.py: `1+torch.zeros(...)`).
    # Overwrite them in-place with iid Uniform(0, 1) samples so the test can catch
    # per-element or mis-mapping of heads bugs that a uniform constant would mask. 
    with torch.no_grad():
        # Cannot assign the model parameter directly; otherwise, the following error will occur:
        # TypeError: cannot assign 'torch.cuda.FloatTensor' as parameter 'B_bias' (torch.nn.Parameter or None expected)
        #model.B_bias = torch.rand_like(model.B_bias)
        #model.C_bias = torch.rand_like(model.C_bias)
        """
        Example for .copy_():
        t = torch.tensor([1.0, 2.0, 3.0])
        print(f"before a = t.copy_(), t = {t}") # tensor([1.0, 2.0, 3.0])
        t.copy_(torch.tensor([4.0, 5.0, 6.0])) # copy tensor([4.0, 5.0, 6.0]) to the memory address of `t`
        print(f"after t.copy_(), t = {t}") # tensor([4.0, 5.0, 6.0])
        """
        # torch.rand_like(model.B_bias) generates Uniform[0, 1) with the same shape/dtype/device as the tensor passed in.
        model.B_bias.copy_(torch.rand_like(model.B_bias))
        model.C_bias.copy_(torch.rand_like(model.C_bias))

    # B_norm and C_norm parameters have all 1's as default values, but we don't know what the typical 
    # distributions are, so keep it unchanged.

    # D parameter has also all 1's as default values, and its values will depend on the bc_head configuration,
    # but we don't know what is the typical distribution of it, so temporary keeping it unchanged

    """
    model.state_dict() creates an ordered dictionary recording each key-value pair of
    the learned parameter tensor of the mamba3 block, for example:
    {
        "in_proj.weight":  tensor(shape=[...]),   # the input projection Linear layer's weight matrix
        "out_proj.weight": tensor(shape=[...]),   # the output projection Linear layer's weight matrix
        "dt_bias":         tensor(shape=[nheads]), # learned per-head bias added before softplus(dt)
        "D":               tensor(shape=[nheads]), # learned per-head skip-connection scale
        "norm.weight":     tensor(shape=[...]),    # the gated RMSNorm's learnable scale
        ...
    }
    """
    # `for k, v in model.state_dict().items()` returns key-value pair of each learned 
    # parameter tensor iteratively
    """
    .detach() returns a view (i.e., not the true data copy) of a tensor without its 
    autograd graph, which is for training, not needed for inference
    """
    # .cpu() moves the data originally on the GPU to the CPU so that we can write the 
    # tensor content to a .pt file later on
    """
    .clone() forces pyTorch to make a true copy for a data, instead of a view
    Example for the difference between with and without .clone():
    import torch

    # Simulate a model parameter that is ALREADY on CPU
    # (e.g. if CONFIG in gen_shared_data.py later added device="cpu",
    # or Mamba3's default device changed)
    v = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    print(v.device)   # cpu

    # --- Your line 38 chain, without .clone() ---
    saved = v.detach().cpu()

    # .cpu() on a tensor that's ALREADY on CPU is a documented no-op:
    # it returns the SAME underlying storage, not a copy.
    print(saved.data_ptr() == v.data_ptr())   # True -- same memory!

    # --- Now the model keeps training / running inference ---
    with torch.no_grad():
        v.add_(100.0)     # some later in-place update to the live model's weight

    print(v)       # tensor([101., 102., 103.])
    print(saved)   # tensor([101., 102., 103.])  <-- "saved" checkpoint silently changed too!

    Because saved shares the exact same memory as the live v, the in-place v.add_(100.0) mutates both — your shared_data.pt would end up serializing corrupted weights (whatever the model happened to look like at the time torch.save actually ran, not at the time you thought you'd frozen it at line 38).

    Now contrast with .clone() added back in:

    saved = v.detach().cpu().clone()
    print(saved.data_ptr() == v.data_ptr())   # False -- independent memory

    with torch.no_grad():
        v.add_(100.0)

    print(v)       # tensor([101., 102., 103.])
    print(saved)   # tensor([1., 2., 3.])  <-- untouched, correctly frozen
    """
    state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # __file__ means the current path this .py file is located
    # os.path.abspath() converts a relative path to an absolute one
    # os.path.dirname() returns only the path name without the file name
    # os.path.join() concates the path with the filename
    """
    Example for os.path.dirname() and os.path.join():
    A = "~/example/example.py"
    A = os.path.dirname(A) # A = "~/example"
    B = "example.txt"
    C = os.path.join(A, B) # C = "~/example/example.txt"
    """
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_data.pt")
    """
    torch.save(obj, path) stores an `obj` Python object to a file at 
    `path` in an optimized way (i.e., pickle protocol)
    """
    # dict_object = {key1: value1, key2: value2, ...}
    torch.save({"input": x.cpu(), "state_dict": state_dict, "config": CONFIG}, out_path)
    # Ctrl+F for the first occurrence of "join", "dirname", and "abspath" to understand them
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_shared_data.log")
    # `with open(log_path, "w") as f:` ensures that the file (i.e., `log_path`) is opened, and
    # closes it automatically after the end of this `with` block
    with open(log_path, "w") as f:
        # f.write() writes a string to the file without appending '\n'
        f.write(f"saved {out_path}\n")
        f.write(f"  input shape: {tuple(x.shape)}, dtype={x.dtype}\n")
        f.write(f"  state_dict keys ({len(state_dict)}): {sorted(state_dict.keys())}\n")

"""
`__name__` is a special built-in variable in Python.
If we run `python gen_shared_data.py` on the terminal directly, 
the value of `__name__` will be "__main__". 
If `gen_shared_data.py` is executed by other files importing it, 
the value of `__name__` will be "gen_shared_data".
"""
# using `if __name__ == "__main__"` is the standard practice to 
# avoid the whole main function being run automatically when importing this .py file
if __name__ == "__main__":
    main()
