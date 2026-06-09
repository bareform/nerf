from typing import Callable

import torch

class PositionalEmbeddings(object):
    def __init__(
        self,
        in_dims: int=3,
        max_freq: int=10,
        num_freqs: int=10,
        include_input: bool=True,
        use_log_sampling: bool=True,
        periodic_functions: tuple[Callable[[torch.Tensor], torch.Tensor], ...]=(torch.sin, torch.cos)
    ) -> None:
        self.in_dims = in_dims
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.use_log_sampling = use_log_sampling
        self.periodic_functions = periodic_functions

        self.embeddings = []
        out_dim = 0
        if self.include_input:
            self.embeddings.append(lambda inputs : inputs)
            out_dim += self.in_dims
        
        if self.use_log_sampling:
            freq_bands = 2.**torch.linspace(0., self.max_freq, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**self.max_freq, steps=self.num_freqs)
        
        freq_bands = freq_bands.tolist()
        for freq in freq_bands:
            for periodic_function in self.periodic_functions:
                self.embeddings.append(
                    lambda inputs, periodic_function=periodic_function, freq=freq:
                        periodic_function(inputs * freq)
                )
                out_dim += self.in_dims
        self.out_dim = out_dim

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.cat(
            [emb(inputs) for emb in self.embeddings],
            dim=-1,
        )
        return outputs
