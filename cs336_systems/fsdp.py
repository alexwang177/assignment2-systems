import torch
import torch.distributed as dist
import torch.nn.functional as F

from einops import einsum

from cs336_basics.model import Embedding, Linear, RMSNorm

class ShardedModule(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, rank: int, world_size: int, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.shape = module.weight.shape
        self.dtype = module.weight.dtype
        self.compute_dtype = compute_dtype
        self.rank = rank
        self.world_size = world_size
        self.padding = self._calculate_padding()
        self.prefetch_handle = None
        self.prefetch_buffer = None
        self.prefetch_fn = None

        self.full_weight = None

        # Shard the weight across ranks initially
        self.shard()

    def forward(self, *inputs, **kwargs):
        # self.full_weight = self.unshard()

        # Wait for current layer i
        self.full_weight = self._wait_gather()

        self.full_weight.requires_grad_(True)
        self.full_weight.retain_grad()
        
        compute_weight = self.full_weight
        if self.compute_dtype:
            compute_weight = self.full_weight.to(self.compute_dtype)

        if isinstance(self.module, Linear):
            out = einsum(inputs[0], compute_weight, "... d_in, d_out d_in -> ... d_out")
        elif isinstance(self.module, Embedding):
            out = compute_weight[inputs[0], :]
        else:
            raise Exception("Unsupported nn.Module")
        
        # Prefetch
        if self.prefetch_fn is not None:
            self.prefetch_fn()

        return out

    def _calculate_padding(self) -> int:
         # Flatten the tensor
        flat_tensor = torch.flatten(self.module.weight)

        # Pad the tensor
        modulo = len(flat_tensor) % self.world_size
        pad = 0
        if modulo > 0:
            desired = (len(flat_tensor) // self.world_size + 1) * self.world_size
            pad = desired - len(flat_tensor)

        return pad

    def _flatten_and_pad(self, t: torch.Tensor) -> torch.Tensor:
        # Flatten the tensor
        flat_tensor = torch.flatten(t)

        # Pad the tensor
        if self.padding > 0:
            return F.pad(flat_tensor, (0, self.padding))
        else:
            return flat_tensor

    def shard(self):
        flat_tensor = self._flatten_and_pad(self.module.weight)

        # Shard the tensor across ranks
        shards = torch.chunk(flat_tensor, self.world_size)
        local_shard = torch.nn.Parameter(shards[self.rank].detach().clone())
        self.module.weight = local_shard

    def _set_prefetch_fn(self, prefetch_fn):
        self.prefetch_fn = prefetch_fn

    def _start_gather(self):
        shard_len = len(self.module.weight)
        self.prefetch_buffer = [torch.zeros((shard_len,), device=self.module.weight.device) for _ in range(self.world_size)]
        self.prefetch_handle = dist.all_gather(self.prefetch_buffer, self.module.weight, async_op=True)

    def _wait_gather(self):

        if self.prefetch_handle is not None:
            self.prefetch_handle.wait()

        assert self.prefetch_buffer is not None
        shard_tensors = self.prefetch_buffer

        # Concatenate
        flat_tensor = torch.cat(shard_tensors)

        # Remove padding
        if self.padding > 0:
            flat_tensor = flat_tensor[:-self.padding]

        # Reshape
        full_weight = torch.reshape(flat_tensor, self.shape)

        # Clear handle and buffer
        self.prefetch_buffer = None
        self.prefetch_handle = None

        return full_weight
    
    def shard_gradient(self):
        flat_grad = self._flatten_and_pad(self.full_weight.grad)

        # Reduce scatter
        shard_len = len(self.module.weight)
        local_grad = torch.zeros((shard_len), device=self.module.weight.device)

        grad_shards = list(torch.chunk(flat_grad, self.world_size))
        dist.reduce_scatter(local_grad, grad_shards, op=dist.ReduceOp.AVG)

        with torch.no_grad():
            self.module.weight.grad = local_grad

class FullyShardedDataParallel(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None, prefetch_layers: int = 2):
        assert dist.is_initialized()

        super().__init__()

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.module = module
        self.compute_dtype = compute_dtype
        self.replicated_params = []
        self.module_idx = 0
        self.module_list = []

        self._build_sharded_modules(self.module)

        # Set prefetch targets
        self.prefetch_layers = prefetch_layers
        for i in range(len(self.module_list) - prefetch_layers):
            target = self.module_list[i+prefetch_layers]
            self.module_list[i]._set_prefetch_fn(target._start_gather)

        # TODO: implement prefetching for backwards too


    def _build_sharded_modules(self, parent):

        for name, child in parent.named_children():
            if isinstance(child, Embedding) or isinstance(child, Linear):
                sharded_module = ShardedModule(child, rank=self.rank, world_size=self.world_size, compute_dtype=self.compute_dtype)
                parent.add_module(name, sharded_module)
                self.module_list.append(sharded_module)
                self.module_idx += 1
            elif isinstance(child, RMSNorm):
                self.replicated_params.extend(child.parameters())
            else:
                self._build_sharded_modules(child)

    def get_unsharded_state_dict(self):

        state_dict = {}
  
        for mod_name, module in self.module.named_modules():

            if isinstance(module, ShardedModule):
                module._start_gather()
                state_dict[f"{mod_name}.weight"] = module._wait_gather()
            elif isinstance(module, RMSNorm):
                for param_name, param in module.named_parameters(recurse=False):
                    state_dict[f"{mod_name}.{param_name}"] = param

        print(state_dict.keys())
        return state_dict

    def forward(self, *inputs, **kwargs):
        # Prefetch the first prefetch layers
        for i in range(self.prefetch_layers):
            self.module_list[i]._start_gather()

        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):

        for _, mod in self.module.named_modules():
            if isinstance(mod, ShardedModule):
                mod.shard_gradient()
                mod.full_weight = None
        
        for p in self.replicated_params:
            if p.grad is None: continue

            with torch.no_grad():
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
