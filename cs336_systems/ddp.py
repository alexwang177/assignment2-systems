import torch
import torch.distributed as dist


class DistributedDataParallel(torch.nn.Module):

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        assert dist.is_initialized()

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.module = module
        self.handles = []

        # broadcast model parameters from rank 0 to all other ranks
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)

                if param.requires_grad:
                    # register post_grad hooks
                    param.register_post_accumulate_grad_hook(self._all_reduce_hook)

    def _all_reduce_hook(self, param: torch.Tensor):
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        self.handles.append(handle)


    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self) -> None:
        for handle in self.handles:
            handle.wait()

        self.handles = []
