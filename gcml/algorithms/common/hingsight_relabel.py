from typing import Iterable

from .constants import Trajectory, Transition


def hindsight_relabel(
        trajectory: Trajectory,
        n_relabeled_transitions: int,
        ordered: bool,
) -> Iterable[Transition]:
    raise NotImplementedError
