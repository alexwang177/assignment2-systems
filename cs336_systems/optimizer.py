import torch
import torch.distributed as dist

from typing import Type, Any

class ShardedOptimizer(torch.optim.Optimizer):

    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        assert dist.is_initialized()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.params = [p for p in params]

        self.rank_params = {rank: [] for rank in range(self.world_size)}
        self.idx = 0
        self.local_optimizer = None

        super().__init__(self.params, dict(**kwargs))

        self.local_optimizer = optimizer_cls(self._get_local_params(), **kwargs)

    def step(self, closure=None, **kwargs): 
        ret = self.local_optimizer.step(closure, **kwargs)

        # broad the local rank's updated params to all other ranks
        for rank in range(self.world_size):
            for p in self.rank_params[rank]:
                dist.broadcast(p, src=rank)

        return ret

    def _get_local_params(self):
        return [p for p in self.rank_params[self.rank]]

    def add_param_group(self, param_group: dict[str, Any]):

        params = [p for p in param_group["params"]]
        param_group.pop("params")

        super().add_param_group({"params": params, **param_group})

        new_params = []

        for p in params:
            if self.idx == self.rank:
                new_params.append(p)

            self.rank_params[self.idx].append(p)
            self.idx = (self.idx + 1) % self.world_size

        if self.local_optimizer is not None:
            self.local_optimizer.add_param_group({"params": new_params, **param_group})
