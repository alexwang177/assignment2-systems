import torch
import torch.distributed as dist

class DistributedDataParallel(torch.nn.Module):

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        assert dist.is_initialized()

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.module = module

        # broadcast model parameters from rank 0 to all other ranks
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.module(input)
