"""Loss utilities shared by RSSM variants."""

import torch.distributions as td
from torch import Tensor


def likelihood(prediction: Tensor, target: Tensor, event_ndims: int, scale: float = 1.0) -> Tensor:
    """Calculate the likelihood of the prediction.

    Args:
        prediction: The prediction.
        target: The target.
        event_ndims: The number of event dimensions.
        scale: The scale of the normal distribution.

    Returns
    -------
    Tensor
        The likelihood of the prediction.
    """
    dist = td.Independent(td.Normal(prediction, scale), event_ndims)
    log_prob: Tensor = dist.log_prob(target)  # type: ignore[no-untyped-call]
    return -log_prob.mean()
