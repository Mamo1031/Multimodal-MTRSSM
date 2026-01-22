"""State utilities shared by RSSM variants."""

from collections.abc import Generator

import torch
from distribution_extension import Distribution
from distribution_extension.utils import cat_distribution, stack_distribution
from torch import Tensor


class State:
    """Latent state with deterministic and stochastic parts."""

    def __init__(self, deter: Tensor, distribution: Distribution, stoch: Tensor | None = None) -> None:
        self.deter = deter
        self.distribution = distribution
        self.stoch = distribution.rsample() if stoch is None else stoch
        self.feature = torch.cat([self.deter, self.stoch], dim=-1)

    def __iter__(self) -> Generator["State", None, None]:
        """Iterate over the state.

        Yields
        ------
        State: The state.
        """
        for i in range(self.deter.shape[0]):
            yield self[i]

    def __getitem__(self, loc: slice | int | tuple[slice | int, ...]) -> "State":
        """Get item from the state.

        Args:
            loc: The location of the item.

        Returns
        -------
        State: The item from the state.
        """
        return type(self)(
            deter=self.deter[loc],
            stoch=self.stoch[loc],
            distribution=self.distribution[loc],
        )

    def to(self, device: torch.device) -> "State":
        """Move the state to the device.

        Args:
            device: The device to move the state to.

        Returns
        -------
        State: The state on the device.
        """
        return type(self)(
            deter=self.deter.to(device),
            stoch=self.stoch.to(device),
            distribution=self.distribution.to(device),
        )

    def detach(self) -> "State":
        """Detach the state.

        Returns
        -------
        State: The detached state.
        """
        return type(self)(
            deter=self.deter.detach(),
            stoch=self.stoch.detach(),
            distribution=self.distribution.detach(),
        )

    def clone(self) -> "State":
        """Clone the state.

        Returns
        -------
        State: The cloned state.
        """
        return type(self)(
            deter=self.deter.clone(),
            stoch=self.stoch.clone(),
            distribution=self.distribution.clone(),
        )

    def squeeze(self, dim: int) -> "State":
        """Squeeze the state.

        Args:
            dim: The dimension to squeeze.

        Returns
        -------
        State: The squeezed state.
        """
        return type(self)(
            deter=self.deter.squeeze(dim),
            stoch=self.stoch.squeeze(dim),
            distribution=self.distribution.squeeze(dim),
        )

    def unsqueeze(self, dim: int) -> "State":
        """Unsqueeze the state.

        Args:
            dim: The dimension to unsqueeze.

        Returns
        -------
        State: The unsqueeze state.
        """
        return type(self)(
            deter=self.deter.unsqueeze(dim),
            stoch=self.stoch.unsqueeze(dim),
            distribution=self.distribution.unsqueeze(dim),
        )


def stack_states(states: list[State], dim: int) -> State:
    """Stack the states.

    Args:
        states: The states to stack.
        dim: The dimension to stack.

    Returns
    -------
    State: The stacked state.
    """
    deter = torch.stack([s.deter for s in states], dim=dim)
    stoch = torch.stack([s.stoch for s in states], dim=dim)
    distribution = stack_distribution([s.distribution for s in states], dim)
    return State(deter=deter, stoch=stoch, distribution=distribution)


def cat_states(states: list[State], dim: int) -> State:
    """Cat the states.

    Args:
        states: The states to cat.
        dim: The dimension to cat.

    Returns
    -------
    State: The cat state.
    """
    deter = torch.cat([s.deter for s in states], dim=dim)
    stoch = torch.cat([s.stoch for s in states], dim=dim)
    distribution = cat_distribution([s.distribution for s in states], dim)
    return State(deter=deter, stoch=stoch, distribution=distribution)
