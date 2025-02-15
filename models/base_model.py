
# Package imports
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def init(self, init_configs):
        pass

    def get_submodels(self):
        raise NotImplemented

    def get_distributed_data_parallel_find_unused_parameters(self):
        # Return True to be safe
        return True
