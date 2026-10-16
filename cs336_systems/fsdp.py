import torch
import torch.distributed as dist
import torch.nn.functional as F

from cs336_basics.model import Embedding, Linear, RMSNorm

class FullyShardedDataParallel(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        assert dist.is_initialized()

        super().__init__()

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.module = module
        self.compute_dtype = compute_dtype

        with torch.no_grad():
            for name, mod in self.module.named_modules():

                if isinstance(mod, Embedding) or isinstance(mod, Linear):
                    t = mod.weight

                    # Flatten the tensor
                    org_shape = t.shape
                    flat_tensor = torch.flatten(t)

                    # Pad the tensor
                    modulo = len(flat_tensor) % self.world_size
                    desired = len(flat_tensor) - modulo + self.world_size
                    pad = desired - len(flat_tensor)

                    if modulo != 0:
                        flat_tensor = F.pad(flat_tensor, (0, pad))

                    # Shard the tensor across ranks
                    shards = torch.chunk(flat_tensor, self.world_size)
                    local_shard = torch.nn.Parameter(shards[self.rank].detach().clone())
                    mod.weight = local_shard



    def forward(self, *inputs, **kwargs):
        pass

    def finish_gradient_synchronization(self):
        pass
