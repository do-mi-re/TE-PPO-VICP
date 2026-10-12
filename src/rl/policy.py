"""Policy-side helpers for the leader PPO implementation.

Exactness note:
- Paper-exact: prices use ``p_k=p_max*sigmoid(z_k)``; bandwidth follows the
  masked sigmoid/softmax allocation in ``docs/manuscript_revision_snippet.tex``.
- Implementation convenience only: the policy helper surfaces SB3
  ``policy_kwargs`` from small config dataclasses so actor and critic settings
  stay explicit in the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from torch import nn

from src.rl.value import CriticNetworkConfig
from src.sim.state import LeaderAction

__all__ = [
    "ActorNetworkConfig",
    "activation_class",
    "build_policy_kwargs",
    "decode_bandwidth",
    "decode_leader_action",
    "decode_prices",
    "latent_action_dim",
    "price_logit_for_fraction",
    "split_latent_action",
    "stable_sigmoid",
]


_MAX_SIGMOID_MAGNITUDE: Final[float] = 60.0


@dataclass(frozen=True)
class ActorNetworkConfig:
    """Configuration for the PPO actor branch."""

    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    share_features_extractor: bool = True
    log_std_init: float = -1.0


def latent_action_dim(num_tasks: int, num_vehicles: int) -> int:
    """Return the raw actor output size ``K + 1 + V``.

    Source: the maintained manuscript equations.
    """

    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    if num_vehicles <= 0:
        raise ValueError("num_vehicles must be positive")
    return num_tasks + 1 + num_vehicles


def split_latent_action(
    raw_action: np.ndarray,
    *,
    num_tasks: int,
    num_vehicles: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Split the latent actor vector into pricing, utilization, and bandwidth logits."""

    action = np.asarray(raw_action, dtype=np.float64)
    expected_dim = latent_action_dim(num_tasks, num_vehicles)
    if action.shape != (expected_dim,):
        raise ValueError("raw_action has the wrong shape")

    price_logits = action[:num_tasks]
    utilization_logit = float(action[num_tasks])
    bandwidth_logits = action[num_tasks + 1 :]
    return price_logits, utilization_logit, bandwidth_logits


def stable_sigmoid(logit: np.ndarray | float) -> np.ndarray | float:
    """Numerically-stable sigmoid for logits that may come from arbitrary actors."""

    clipped = np.clip(logit, -_MAX_SIGMOID_MAGNITUDE, _MAX_SIGMOID_MAGNITUDE)
    return 1.0 / (1.0 + np.exp(-clipped))


def price_logit_for_fraction(fraction: float) -> float:
    """Return the price-head bias whose sigmoid equals ``fraction``.

    Exactness label: implementation convenience only.  The bias changes the
    initial exploration distribution but neither the sigmoid mapping nor the
    feasible interval ``[0, p_max]``.
    """

    if not 0.0 < fraction < 1.0:
        raise ValueError("initial price fraction must lie strictly between 0 and 1")
    return float(np.log(fraction / (1.0 - fraction)))


def decode_prices(price_logits: np.ndarray, *, pricing_budget_max: float) -> np.ndarray:
    """Decode pricing logits to ``[0, p_max]``.

    Source: the maintained manuscript equations.
    """

    if pricing_budget_max <= 0.0:
        raise ValueError("pricing_budget_max must be positive")
    logits = np.asarray(price_logits, dtype=np.float64)
    return pricing_budget_max * np.asarray(stable_sigmoid(logits), dtype=np.float64)


def decode_bandwidth(
    utilization_logit: float,
    bandwidth_logits: np.ndarray,
    *,
    eligible_vehicle_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Decode bandwidth logits over vehicles eligible after candidate pruning.

    Paper-exact: for the vehicle-level union mask :math:`m_i`, this implements
    ``w_i = u m_i exp(z_i) / sum_j m_j exp(z_j)``.  With no mask, every
    vehicle is eligible and the expression reduces to the unmasked softmax.
    An empty eligible set receives zero bandwidth.
    """

    logits = np.asarray(bandwidth_logits, dtype=np.float64)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("bandwidth_logits must be a non-empty 1D array")
    if not np.all(np.isfinite(logits)):
        raise ValueError("bandwidth_logits must be finite")

    if eligible_vehicle_mask is None:
        eligible = np.ones(logits.shape, dtype=bool)
    else:
        raw_mask = np.asarray(eligible_vehicle_mask)
        if raw_mask.shape != logits.shape:
            raise ValueError("eligible_vehicle_mask has the wrong shape")
        if not np.all(np.isfinite(raw_mask)):
            raise ValueError("eligible_vehicle_mask must be finite")
        if not np.all((raw_mask == 0) | (raw_mask == 1)):
            raise ValueError("eligible_vehicle_mask must contain only 0/1 values")
        eligible = raw_mask.astype(bool)

    if not np.isfinite(utilization_logit):
        raise ValueError("utilization_logit must be finite")
    utilization = float(stable_sigmoid(float(utilization_logit)))
    proportions = np.zeros_like(logits)
    if not np.any(eligible):
        return proportions
    eligible_logits = logits[eligible]
    centered = eligible_logits - float(np.max(eligible_logits))
    eligible_proportions = np.exp(centered)
    eligible_proportions /= float(np.sum(eligible_proportions))
    proportions[eligible] = eligible_proportions
    return utilization * proportions


def decode_leader_action(
    raw_action: np.ndarray,
    *,
    num_tasks: int,
    num_vehicles: int,
    pricing_budget_max: float,
    eligible_vehicle_mask: np.ndarray | None = None,
) -> LeaderAction:
    """Decode the PPO latent action into the paper's leader action tuple."""

    price_logits, utilization_logit, bandwidth_logits = split_latent_action(
        raw_action,
        num_tasks=num_tasks,
        num_vehicles=num_vehicles,
    )
    prices = decode_prices(price_logits, pricing_budget_max=pricing_budget_max)
    bandwidth = decode_bandwidth(
        utilization_logit,
        bandwidth_logits,
        eligible_vehicle_mask=eligible_vehicle_mask,
    )
    return LeaderAction(
        prices=tuple(float(price) for price in prices),
        bandwidth=tuple(float(weight) for weight in bandwidth),
    )


def activation_class(name: str) -> type[nn.Module]:
    """Map a small string config to a torch activation class."""

    lowered = name.lower()
    if lowered == "relu":
        return nn.ReLU
    if lowered == "tanh":
        return nn.Tanh
    if lowered == "elu":
        return nn.ELU
    raise ValueError(f"unsupported activation: {name}")


def build_policy_kwargs(
    *,
    actor: ActorNetworkConfig,
    critic: CriticNetworkConfig,
) -> dict[str, object]:
    """Build explicit SB3 ``policy_kwargs`` with separate actor and critic branches."""

    activation = activation_class(actor.activation)
    if actor.activation.lower() != critic.activation.lower():
        raise ValueError("actor and critic activations must currently match")
    return {
        "activation_fn": activation,
        "net_arch": {
            "pi": list(actor.hidden_sizes),
            "vf": list(critic.hidden_sizes),
        },
        "share_features_extractor": actor.share_features_extractor,
        "log_std_init": actor.log_std_init,
    }
