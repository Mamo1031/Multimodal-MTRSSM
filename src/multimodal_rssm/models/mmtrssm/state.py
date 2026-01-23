"""State utilities for Multimodal MTRSSM (hierarchical states)."""

from collections.abc import Generator

import torch
from distribution_extension import Distribution
from distribution_extension.utils import cat_distribution, stack_distribution
from torch import Tensor


class MTState:
    """Hierarchical latent state with higher and lower layers.

    Each layer has deterministic and stochastic parts:
    - Higher layer: deter_h (deterministic), stoch_h (stochastic)
    - Lower layer: deter_l (deterministic), stoch_l (stochastic)
    """

    def __init__(
        self,
        deter_h: Tensor,
        deter_l: Tensor,
        distribution_h: Distribution,
        distribution_l: Distribution,
        hidden_h: Tensor,
        hidden_l: Tensor,
        stoch_h: Tensor | None = None,
        stoch_l: Tensor | None = None,
    ) -> None:
        """Initialize MTState.

        Args:
            deter_h: Higher layer deterministic state.
            deter_l: Lower layer deterministic state.
            distribution_h: Higher layer probability distribution.
            distribution_l: Lower layer probability distribution.
            hidden_h: Higher layer RNN hidden state.
            hidden_l: Lower layer RNN hidden state.
            stoch_h: Higher layer stochastic state (sampled if None).
            stoch_l: Lower layer stochastic state (sampled if None).
        """
        self.deter_h = deter_h
        self.deter_l = deter_l
        self.distribution_h = distribution_h
        self.distribution_l = distribution_l
        self.hidden_h = hidden_h
        self.hidden_l = hidden_l
        self.stoch_h = distribution_h.rsample() if stoch_h is None else stoch_h
        self.stoch_l = distribution_l.rsample() if stoch_l is None else stoch_l
        # Feature: concatenate all states [h_deter, h_stoch, l_deter, l_stoch]
        self.feature = torch.cat([self.deter_h, self.stoch_h, self.deter_l, self.stoch_l], dim=-1)

    def __iter__(self) -> Generator["MTState", None, None]:
        """Iterate over the state.

        Yields
        ------
        MTState: The state.
        """
        for i in range(self.deter_h.shape[0]):
            yield self[i]

    def __getitem__(self, loc: slice | int | tuple[slice | int, ...]) -> "MTState":
        """Get item from the state.

        Args:
            loc: The location of the item.

        Returns
        -------
        MTState: The item from the state.
        """
        return type(self)(
            deter_h=self.deter_h[loc],
            deter_l=self.deter_l[loc],
            distribution_h=self.distribution_h[loc],
            distribution_l=self.distribution_l[loc],
            hidden_h=self.hidden_h[loc] if self.hidden_h.dim() > 1 else self.hidden_h,
            hidden_l=self.hidden_l[loc] if self.hidden_l.dim() > 1 else self.hidden_l,
            stoch_h=self.stoch_h[loc],
            stoch_l=self.stoch_l[loc],
        )

    def to(self, device: torch.device) -> "MTState":
        """Move the state to the device.

        Args:
            device: The device to move the state to.

        Returns
        -------
        MTState: The state on the device.
        """
        return type(self)(
            deter_h=self.deter_h.to(device),
            deter_l=self.deter_l.to(device),
            distribution_h=self.distribution_h.to(device),
            distribution_l=self.distribution_l.to(device),
            hidden_h=self.hidden_h.to(device),
            hidden_l=self.hidden_l.to(device),
            stoch_h=self.stoch_h.to(device),
            stoch_l=self.stoch_l.to(device),
        )

    def detach(self) -> "MTState":
        """Detach the state.

        Returns
        -------
        MTState: The detached state.
        """
        return type(self)(
            deter_h=self.deter_h.detach(),
            deter_l=self.deter_l.detach(),
            distribution_h=self.distribution_h.detach(),
            distribution_l=self.distribution_l.detach(),
            hidden_h=self.hidden_h.detach(),
            hidden_l=self.hidden_l.detach(),
            stoch_h=self.stoch_h.detach(),
            stoch_l=self.stoch_l.detach(),
        )

    def clone(self) -> "MTState":
        """Clone the state.

        Returns
        -------
        MTState: The cloned state.
        """
        return type(self)(
            deter_h=self.deter_h.clone(),
            deter_l=self.deter_l.clone(),
            distribution_h=self.distribution_l.clone(),
            distribution_l=self.distribution_l.clone(),
            hidden_h=self.hidden_h.clone(),
            hidden_l=self.hidden_l.clone(),
            stoch_h=self.stoch_h.clone(),
            stoch_l=self.stoch_l.clone(),
        )

    def squeeze(self, dim: int) -> "MTState":
        """Squeeze the state.

        Args:
            dim: The dimension to squeeze.

        Returns
        -------
        MTState: The squeezed state.
        """
        return type(self)(
            deter_h=self.deter_h.squeeze(dim),
            deter_l=self.deter_l.squeeze(dim),
            distribution_h=self.distribution_h.squeeze(dim),
            distribution_l=self.distribution_l.squeeze(dim),
            hidden_h=self.hidden_h.squeeze(dim) if self.hidden_h.dim() > 1 else self.hidden_h,
            hidden_l=self.hidden_l.squeeze(dim) if self.hidden_l.dim() > 1 else self.hidden_l,
            stoch_h=self.stoch_h.squeeze(dim),
            stoch_l=self.stoch_l.squeeze(dim),
        )

    def unsqueeze(self, dim: int) -> "MTState":
        """Unsqueeze the state.

        Args:
            dim: The dimension to unsqueeze.

        Returns
        -------
        MTState: The unsqueezed state.
        """
        return type(self)(
            deter_h=self.deter_h.unsqueeze(dim),
            deter_l=self.deter_l.unsqueeze(dim),
            distribution_h=self.distribution_h.unsqueeze(dim),
            distribution_l=self.distribution_l.unsqueeze(dim),
            hidden_h=self.hidden_h.unsqueeze(dim) if self.hidden_h.dim() > 1 else self.hidden_h,
            hidden_l=self.hidden_l.unsqueeze(dim) if self.hidden_l.dim() > 1 else self.hidden_l,
            stoch_h=self.stoch_h.unsqueeze(dim),
            stoch_l=self.stoch_l.unsqueeze(dim),
        )


def stack_mtstates(states: list[MTState], dim: int) -> MTState:
    """Stack the MTStates.

    Args:
        states: The states to stack.
        dim: The dimension to stack.

    Returns
    -------
    MTState: The stacked state.
    """
    deter_h = torch.stack([s.deter_h for s in states], dim=dim)
    deter_l = torch.stack([s.deter_l for s in states], dim=dim)
    stoch_h = torch.stack([s.stoch_h for s in states], dim=dim)
    stoch_l = torch.stack([s.stoch_l for s in states], dim=dim)
    distribution_h = stack_distribution([s.distribution_h for s in states], dim)
    distribution_l = stack_distribution([s.distribution_l for s in states], dim)
    hidden_h = (
        torch.stack([s.hidden_h for s in states], dim=dim) if states[0].hidden_h.dim() > 1 else states[0].hidden_h
    )
    hidden_l = (
        torch.stack([s.hidden_l for s in states], dim=dim) if states[0].hidden_l.dim() > 1 else states[0].hidden_l
    )
    return MTState(
        deter_h=deter_h,
        deter_l=deter_l,
        distribution_h=distribution_h,
        distribution_l=distribution_l,
        hidden_h=hidden_h,
        hidden_l=hidden_l,
        stoch_h=stoch_h,
        stoch_l=stoch_l,
    )


def cat_mtstates(states: list[MTState], dim: int) -> MTState:
    """Concatenate the MTStates.

    Args:
        states: The states to concatenate.
        dim: The dimension to concatenate.

    Returns
    -------
    MTState: The concatenated state.
    """
    deter_h = torch.cat([s.deter_h for s in states], dim=dim)
    deter_l = torch.cat([s.deter_l for s in states], dim=dim)
    stoch_h = torch.cat([s.stoch_h for s in states], dim=dim)
    stoch_l = torch.cat([s.stoch_l for s in states], dim=dim)
    distribution_h = cat_distribution([s.distribution_h for s in states], dim)
    distribution_l = cat_distribution([s.distribution_l for s in states], dim)
    # For hidden states, use the last state's hidden (they should be the same after rollout)
    hidden_h = states[-1].hidden_h
    hidden_l = states[-1].hidden_l
    return MTState(
        deter_h=deter_h,
        deter_l=deter_l,
        distribution_h=distribution_h,
        distribution_l=distribution_l,
        hidden_h=hidden_h,
        hidden_l=hidden_l,
        stoch_h=stoch_h,
        stoch_l=stoch_l,
    )
