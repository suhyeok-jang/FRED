from abc import ABC, abstractmethod
from typing import Any


class BasePolicy(ABC):
    """Policy interface: observation/action validation, action computation and reset.

    ``strict`` controls whether get_action() runs check_observation()/check_action().
    """

    def __init__(self, *, strict: bool = True):
        self.strict = strict

    @abstractmethod
    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate the observation; raises AssertionError when it is invalid."""
        pass

    @abstractmethod
    def check_action(self, action: dict[str, Any]) -> None:
        """Validate the action; raises AssertionError when it is invalid."""
        pass

    @abstractmethod
    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Policy-specific (action, info) computation; validation is done by the public get_action()."""
        pass

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Validate the observation (when strict), run _get_action(), validate the action (when strict)."""
        if self.strict:
            self.check_observation(observation)
        action, info = self._get_action(observation, options)
        if self.strict:
            self.check_action(action)
        return action, info

    @abstractmethod
    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy; returns an info dict."""
        pass
