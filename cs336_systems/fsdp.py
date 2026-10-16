import torch
import torch.distributed as dist
import torch.nn.functional as F

from einops import einsum, rearrange

from cs336_basics.model import Embedding, Linear, RMSNorm

class ShardedModule(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, rank: int, world_size: int):
        super().__init__()
        self.module = module
        self.shape = module.weight.shape
        self.dtype = module.weight.dtype
        self.rank = rank
        self.world_size = world_size
        self.padding = self._calculate_padding()

        self.full_weight = None

        # Shard the weight across ranks initially
        self.shard()

    def forward(self, *inputs, **kwargs):
        self.full_weight = self.unshard()
        self.full_weight.requires_grad_(True)
        self.full_weight.retain_grad()

        # TODO delegate forward
        if isinstance(self.module, Linear):
            return einsum(inputs[0], self.full_weight, "... d_in, d_out d_in -> ... d_out")
        elif isinstance(self.module, Embedding):
            return self.full_weight[inputs[0], :]
        else:
            raise Exception("Unsupported nn.Module")

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

    def unshard(self):
        shard_len = len(self.module.weight)
        shard_tensors = [torch.zeros((shard_len,), device=self.module.weight.device) for _ in range(self.world_size)]

        dist.all_gather(shard_tensors, self.module.weight)

        # Concatenate
        flat_tensor = torch.cat(shard_tensors)

        # Remove padding
        if self.padding > 0:
            flat_tensor = flat_tensor[:-self.padding]

        # Reshape
        full_weight = torch.reshape(flat_tensor, self.shape)
        return full_weight
    
    def shard_gradient(self):
        flat_grad = self._flatten_and_pad(self.full_weight.grad)

        # Reduce scatter
        shard_len = len(self.module.weight)
        local_grad = torch.zeros((shard_len), device=self.module.weight.device)

        grad_shards = torch.chunk(flat_grad, self.world_size)
        dist.reduce_scatter(local_grad, grad_shards, op=dist.ReduceOp.AVG)

        with torch.no_grad():
            self.module.weight.grad = local_grad

class FullyShardedDataParallel(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        assert dist.is_initialized()

        super().__init__()

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.module = module
        self.compute_dtype = compute_dtype
        self.replicated_params = []

        self._build_sharded_modules(self.module)

    def _build_sharded_modules(self, parent):

        for name, child in parent.named_children():
            if isinstance(child, Embedding) or isinstance(child, Linear):
                parent.add_module(name, ShardedModule(child, rank=self.rank, world_size=self.world_size))
            elif isinstance(child, RMSNorm):
                self.replicated_params.extend(child.parameters())
            else:
                self._build_sharded_modules(child)

    def _forward_pre_hook(self, module, input) -> None:
        # Wait for all-gather to finish
        pass

    def _forward_post_hook(self, module, input, output) -> None:
        # Prefetch weights for layer i+2
        pass


    def forward(self, *inputs, **kwargs):
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
