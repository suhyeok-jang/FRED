from dataclasses import dataclass
import os

import torch
import tyro

from fred.data.embodiment_tags import EmbodimentTag
from fred.policy.fred_policy import FredPolicy, FredSimPolicyWrapper
from fred.policy.server_client import PolicyServer

DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class ServerConfig:
    """Serve a fred checkpoint as a policy server (ZMQ) for the sim rollout client (rollout_policy.py)."""

    model_path: str
    """Checkpoint directory (the EMA weights are loaded)."""
    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    device: str = "cuda"
    host: str = "127.0.0.1"
    port: int = DEFAULT_MODEL_SERVER_PORT
    strict: bool = True
    """Validate every observation and action against the modality config."""
    text_embed_cache: str | None = None
    """wan22_5b: raw (un-projected) umT5 text-embedding cache; the text encoder is not loaded when every prompt hits it."""
    text_embed_require_hit: bool = True
    action_seed: int | None = None
    """Seed of the action-noise generator (paired evals)."""
    video_seed: int | None = None
    """torch.manual_seed before the model is built: fixes the future-latent noise stream."""


def main(config: ServerConfig):
    print("Starting fred policy server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    if config.video_seed is not None:
        torch.manual_seed(int(config.video_seed))
        print(f"  video noise seed: {config.video_seed}")
    print(f"  text cache: {config.text_embed_cache} (require_hit={config.text_embed_require_hit})  action_seed={config.action_seed}")
    assert not (config.model_path.startswith("/") and not os.path.exists(config.model_path)), f"Model path {config.model_path} does not exist"

    policy = FredSimPolicyWrapper(
        FredPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            text_embed_cache=config.text_embed_cache,
            text_embed_require_hit=config.text_embed_require_hit,
            action_seed=config.action_seed,
            strict=config.strict,
        ),
        strict=config.strict,
    )
    server = PolicyServer(policy=policy, host=config.host, port=config.port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
