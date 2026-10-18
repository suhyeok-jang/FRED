"""ZMQ REQ/REP transport between the policy server (fred/eval/serve_policy.py) and the rollout client."""

import io
from typing import Any, Callable

import msgpack
import numpy as np
import zmq

from fred.data.types import ModalityConfig
from fred.data.utils import to_json_serializable

from .policy import BasePolicy


class MsgSerializer:
    """msgpack with np.ndarray (npy bytes) and ModalityConfig (json) hooks."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            return ModalityConfig(**obj["as_json"])
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, ModalityConfig):
            return {"__ModalityConfig_class__": True, "as_json": to_json_serializable(obj)}
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class PolicyServer:
    """REP loop: each request names an endpoint and carries its kwargs; the handler's return value is the reply."""

    def __init__(self, policy: BasePolicy, host: str = "*", port: int = 5555):
        self.policy = policy
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, Callable] = {
            "get_action": policy.get_action,
            "reset": policy.reset,
            "get_modality_config": policy.get_modality_config,
        }

    def run(self):
        print(f"Server is ready and listening on {self.socket.getsockopt_string(zmq.LAST_ENDPOINT)}")
        while True:
            try:
                request = MsgSerializer.from_bytes(self.socket.recv())
                endpoint = request["endpoint"]
                assert endpoint in self._endpoints, f"Unknown endpoint: {endpoint}"
                result = self._endpoints[endpoint](**request["data"])
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as e:
                # flush=True is load-bearing: without it the traceback sits in the stdout
                # buffer and is LOST when the server is torn down, leaving server.log with
                # no record at all while the client shows only `str(e)`. That is how a run
                # of CUDNN_STATUS_INTERNAL_ERROR failures stayed undiagnosable.
                print(f"Error in server: {e}", flush=True)
                import traceback

                print(traceback.format_exc(), flush=True)
                # cuDNN reports a failed workspace allocation as INTERNAL_ERROR rather than a
                # clean OOM, and that path never goes through PyTorch's allocator — so the
                # exception text alone cannot distinguish "real cuDNN fault" from "out of
                # memory". Dump both the device and the allocator view to settle it.
                try:
                    import torch

                    if torch.cuda.is_available():
                        free_b, total_b = torch.cuda.mem_get_info()
                        gb = 1024**3
                        print(
                            f"[server-oom-probe] device={torch.cuda.current_device()} "
                            f"free={free_b / gb:.2f}GiB total={total_b / gb:.2f}GiB "
                            f"torch_allocated={torch.cuda.memory_allocated() / gb:.2f}GiB "
                            f"torch_reserved={torch.cuda.memory_reserved() / gb:.2f}GiB "
                            f"max_reserved={torch.cuda.max_memory_reserved() / gb:.2f}GiB",
                            flush=True,
                        )
                except Exception as _probe_err:  # never let diagnostics mask the real error
                    print(f"[server-oom-probe] unavailable: {_probe_err}", flush=True)
                self.socket.send(MsgSerializer.to_bytes({"error": str(e)}))


class PolicyClient(BasePolicy):
    """REQ client of PolicyServer. Validation runs on the server, so the client is never strict."""

    def __init__(self, host: str = "localhost", port: int = 5555):
        super().__init__(strict=False)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")

    def call_endpoint(self, endpoint: str, data: dict | None = None) -> Any:
        """Send one request to ``endpoint`` and return its decoded response; raises on a server-side error."""
        self.socket.send(MsgSerializer.to_bytes({"endpoint": endpoint, "data": data or {}}))
        response = MsgSerializer.from_bytes(self.socket.recv())
        assert not (isinstance(response, dict) and "error" in response), f"Server error: {response['error']}"
        return response

    def __del__(self):
        self.socket.close()
        self.context.term()

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint("get_action", {"observation": observation, "options": options})
        return tuple(response)  # Convert list (from msgpack) to tuple of (action, info)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config")

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Server-side (strict=False)."""

    def check_action(self, action: dict[str, Any]) -> None:
        """Server-side (strict=False)."""
