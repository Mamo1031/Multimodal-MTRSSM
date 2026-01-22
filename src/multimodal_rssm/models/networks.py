"""Neural networks shared by RSSM variants."""

from __future__ import annotations

import torch
from distribution_extension import MultiOneHotFactory
from torch import Tensor, nn
from torchrl.modules import MLP

from multimodal_rssm.models.state import State

__all__ = [
    "Representation",
    "Transition",
]


class Representation(nn.Module):
    """Representation network for the RSSM."""

    def __init__(
        self,
        *,
        deterministic_size: int,
        hidden_size: int,
        obs_embed_size: int,
        distribution_config: tuple[int, int] | list[int],
        activation_name: str = "ReLU",
    ) -> None:
        """Initialize the representation network.

        Args:
            deterministic_size: Size of deterministic state.
            hidden_size: Size of hidden layer.
            obs_embed_size: Size of observation embedding.
            distribution_config: Tuple of (class_size, category_size).
            activation_name: Name of activation function.

        Raises
        ------
        ValueError
            If distribution_config list does not have exactly 2 elements.
        """
        super().__init__()
        # Convert list to tuple if needed (for YAML config compatibility)
        if isinstance(distribution_config, list):
            distribution_config_tuple: tuple[int, ...] = tuple(distribution_config)
            expected_length = 2
            if len(distribution_config_tuple) != expected_length:
                msg = f"distribution_config must have {expected_length} elements, got {len(distribution_config_tuple)}"
                raise ValueError(msg)
            distribution_config_final: tuple[int, int] = (distribution_config_tuple[0], distribution_config_tuple[1])
        else:
            distribution_config_final = distribution_config
        class_size, category_size = distribution_config_final

        self.rnn_to_post_projector = MLP(
            in_features=obs_embed_size + deterministic_size,
            out_features=class_size * category_size,
            num_cells=hidden_size,
            depth=1,
            activation_class=getattr(torch.nn, activation_name),
            activate_last_layer=False,
        )
        self.distribution_factory = MultiOneHotFactory(
            class_size=class_size,
            category_size=category_size,
        )

    def forward(self, obs_embed: Tensor, prior_state: State) -> State:
        """Forward pass of the representation network.

        Args:
            obs_embed: The observation embedding.
            prior_state: The prior state.

        Returns
        -------
        State: The posterior state.
        """
        projector_input = torch.cat([prior_state.deter, obs_embed], -1)
        stoch_source = self.rnn_to_post_projector(projector_input)
        distribution = self.distribution_factory.forward(stoch_source)
        return State(deter=prior_state.deter, distribution=distribution)


class Transition(nn.Module):
    """Transition network for the RSSM."""

    def __init__(
        self,
        *,
        deterministic_size: int,
        hidden_size: int,
        action_size: int,
        distribution_config: tuple[int, int] | list[int],
        activation_name: str,
    ) -> None:
        """Initialize the transition network.

        Args:
            deterministic_size: Size of deterministic state.
            hidden_size: Size of hidden layer.
            action_size: Size of action.
            distribution_config: Tuple of (class_size, category_size).
            activation_name: Name of activation function.

        Raises
        ------
        ValueError
            If distribution_config list does not have exactly 2 elements.
        """
        super().__init__()
        # Convert list to tuple if needed (for YAML config compatibility)
        if isinstance(distribution_config, list):
            distribution_config_tuple: tuple[int, ...] = tuple(distribution_config)
            expected_length = 2
            if len(distribution_config_tuple) != expected_length:
                msg = f"distribution_config must have {expected_length} elements, got {len(distribution_config_tuple)}"
                raise ValueError(msg)
            distribution_config_final: tuple[int, int] = (distribution_config_tuple[0], distribution_config_tuple[1])
        else:
            distribution_config_final = distribution_config
        class_size, category_size = distribution_config_final

        self.rnn_cell = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=deterministic_size,
        )
        self.action_state_projector = MLP(
            in_features=action_size + class_size * category_size,
            out_features=hidden_size,
            num_cells=hidden_size,
            depth=1,
            activation_class=getattr(torch.nn, activation_name),
            activate_last_layer=False,
        )
        self.rnn_to_prior_projector = MLP(
            in_features=deterministic_size,
            out_features=class_size * category_size,
            num_cells=hidden_size,
            depth=1,
            activation_class=getattr(torch.nn, activation_name),
            activate_last_layer=False,
        )
        self.distribution_factory = MultiOneHotFactory(
            class_size=class_size,
            category_size=category_size,
        )

    def forward(self, action: Tensor, prev_state: State) -> State:
        """Forward pass of the transition network.

        Args:
            action: The action.
            prev_state: The previous state.

        Returns
        -------
        State: The posterior state.
        """
        # stoch is already flattened: (batch, class_size * category_size)
        # If it's not flattened, flatten it for compatibility
        min_dim_for_flatten = 3
        stoch_flat = (
            prev_state.stoch.flatten(start_dim=1) if prev_state.stoch.dim() >= min_dim_for_flatten else prev_state.stoch
        )
        projector_input = torch.cat([action, stoch_flat], dim=-1)
        action_state = self.action_state_projector(projector_input)
        deter = self.rnn_cell.forward(action_state, hx=prev_state.deter)
        stoch_source = self.rnn_to_prior_projector(deter)
        distribution = self.distribution_factory.forward(stoch_source)
        return State(deter=deter, distribution=distribution)
