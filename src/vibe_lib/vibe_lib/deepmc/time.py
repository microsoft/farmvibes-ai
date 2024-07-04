from torch import Tensor, nn


class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first: bool = False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x: Tensor):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        # COMMENT: Can use rearrange here :)
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
