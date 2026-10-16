import torch
import torch.distributed as dist

from typing import Type, Any

class ShardedOptimizer(torch.optim.Optimizer):

    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        assert dist.is_initialized()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        materialized_params = [p for p in params]
        assert len(materialized_params) > 0
        
        if isinstance(materialized_params[0], dict):
            # param groups
            self.param_groups = [p.copy() for p in materialized_params]
        else:
            # normal param
            self.param_groups = [{"params": materialized_params.copy()}]

        self.rank_param_groups = {rank: [] for rank in range(self.world_size)}
        self.idx = 0
        self.local_optimizer = None

        super().__init__(self.param_groups, dict(**kwargs))

        self.local_optimizer = optimizer_cls(self._get_local_param_groups(), **kwargs)

    def step(self, closure=None, **kwargs): 
        ret = self.local_optimizer.step(closure, **kwargs)

        # broadcast the local rank's updated params to all other ranks
        for rank in range(self.world_size):
            for pg in self.rank_param_groups[rank]:
                for p in pg["params"]:
                    with torch.no_grad():
                        dist.broadcast(p, src=rank)

        return ret

    def _get_local_param_groups(self):
        return [pg for pg in self.rank_param_groups[self.rank]]

    def add_param_group(self, param_group: dict[str, Any]):

        params = [p for p in param_group["params"]]
        options = {k: v for k, v in param_group.items() if k != "params"}

        super().add_param_group({"params": params, **options})

        new_param_groups = {rank: None for rank in range(self.world_size)}
        new_params = {rank: [] for rank in range(self.world_size)}

        for p in params:
            new_params[self.idx].append(p)
            self.idx = (self.idx + 1) % self.world_size

        for rank, params in new_params.items():
            new_param_groups[rank] = {"params": params, **self.defaults, **options}
            self.rank_param_groups[rank].append(new_param_groups[rank])

        if not new_params:
            return

        if self.local_optimizer is not None:
            self.local_optimizer.add_param_group(new_param_groups[self.rank])
