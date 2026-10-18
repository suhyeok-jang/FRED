"""Per-episode mp4 (and optional PNG frame) recording around a flat-key sim env, with deterministic per-episode
reset seeding."""

from pathlib import Path
import random
import re
import shutil

import av
import cv2
import gymnasium as gym
import numpy as np
from PIL import Image as PILImage

_VIDEO_RES_TAG_RE = re.compile(r"res(\d+)")


def dedup_resolution_duplicate_video_keys(keys: list[str]) -> list[str]:
    """Drop higher-resolution duplicate streams of the same camera (recording only).

    RoboCasa Kitchen (GrootRoboCasaEnv + PandaOmronKeyConverter) emits both
    ``video.res256_image_{side_0,side_1,wrist_0}`` (the model inputs) AND raw ``video.res512_image_*`` copies of
    the same cameras, which would give a 6-panel mp4 instead of the 3-view recording.

    Rule (name-based): a key is dropped only when another key exists that is identical except for a strictly
    smaller ``res<N>`` tag. Keys without a ``res<N>`` tag or without such a twin are kept, so envs with no
    resolution duplicates (gr1_tabletop / libero) record every camera. Returns the kept keys in input order.
    """
    order = {key: idx for idx, key in enumerate(keys)}
    ungrouped: list[str] = []
    groups: dict[str, list[tuple[int, str]]] = {}
    for key in keys:
        match = _VIDEO_RES_TAG_RE.search(key)
        if match is None:
            ungrouped.append(key)
            continue
        # Same camera at different resolutions differs only in the res tag.
        template = key[: match.start()] + "res*" + key[match.end():]
        groups.setdefault(template, []).append((int(match.group(1)), key))

    kept = ungrouped
    for members in groups.values():
        min_res = min(res for res, _ in members)
        kept.extend(key for res, key in members if res == min_res)
    return sorted(kept, key=order.__getitem__)


def keep_canonical_video_obs(obs: dict) -> dict:
    """obs with duplicate higher-res video streams dropped (see dedup_resolution_duplicate_video_keys); the PNG
    dump then writes the same views as the mp4."""
    video_keys = [k for k in obs if "video" in k]
    kept = set(dedup_resolution_duplicate_video_keys(video_keys))
    return {k: v for k, v in obs.items() if ("video" not in k) or (k in kept)}


class VideoRecorder:
    """PyAV h264 writer: rgb24 frames in, yuv420p mp4 out, one file per start()/stop()."""

    def __init__(self, fps: int, crf: int = 22):
        self.fps = fps
        self.crf = crf
        self.container = None
        self.stream = None
        self.shape = None

    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.stream is not None

    def start(self, file_path: Path):
        if self.is_ready():
            self.stop()
        self.container = av.open(str(file_path), mode="w")
        self.stream = self.container.add_stream("h264", rate=self.fps)
        codec_context = self.stream.codec_context
        codec_context.pix_fmt = "yuv420p"
        codec_context.options = {"crf": str(self.crf), "profile:v": "high"}
        codec_context.thread_type = "FRAME"
        codec_context.thread_count = 1

    def write_frame(self, img: np.ndarray):
        assert self.is_ready(), "Must run start() before writing!"
        if self.shape is None:
            self.shape = img.shape
            h, w, _ = img.shape
            self.stream.width = w
            self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == np.uint8
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def stop(self):
        if not self.is_ready():
            return
        for packet in self.stream.encode():  # flush
            self.container.mux(packet)
        self.container.close()
        self.container = None
        self.stream = None
        self.shape = None


class VideoRecordingWrapper(gym.Wrapper):
    """Records every ``video.*`` observation (canonical views, side by side, instruction + success overlaid)
    to ``output_dir/<name_prefix>-episode_<id>.mp4``; at the next reset() the file is renamed with the outcome
    (``-success`` / ``-failure``) or removed when the episode did not run to ``max_episode_steps``.

    pending_episode_ids: the episode ids this env records, in order (the rollout's resume computes them from
    the mp4s already on disk); once they are exhausted later episodes write nothing. Episode ``k`` is reset
    with seed ``base_seed * seed_stride + k - 1``, so a refilled id reproduces the original run's initial
    conditions. save_frames: also dump per-step PNGs (``video_modality_keys`` = camera suffixes without
    ``video.``; None = every video.* key) plus instruction.txt in ``saved_frames/<mp4 stem>/``.
    """

    def __init__(
        self,
        env,
        output_dir: Path,
        name_prefix: str,
        pending_episode_ids: list[int],
        base_seed: int,
        max_episode_steps: int,
        seed_stride: int = 100000,
        fps: int = 20,
        save_frames: bool = False,
        video_modality_keys: list[str] | None = None,
    ):
        super().__init__(env)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.name_prefix = name_prefix
        self._pending_episode_ids = [int(e) for e in pending_episode_ids]
        self._pending_cursor = 0
        self.base_seed = int(base_seed)
        self.seed_stride = int(seed_stride)
        self.max_episode_steps = max_episode_steps
        self.video_recorder = VideoRecorder(fps=fps)
        self.save_frames = save_frames
        self.video_modality_keys = list(video_modality_keys) if video_modality_keys is not None else None

        self.file_path: Path | None = None
        self.frames_dir: Path | None = None
        self.step_count = 0
        self.is_success = False

    def _resize_frames_to_common_height(self, frames):
        """Resize to the smallest height for horizontal concatenation; even sizes for H.264."""
        target_height = min(frame.shape[0] for frame in frames)
        target_height -= target_height % 2
        resized = []
        for frame in frames:
            if frame.shape[0] != target_height:
                h, w = frame.shape[:2]
                new_width = int(w * target_height / h)
                new_width -= new_width % 2
                frame = cv2.resize(frame, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized.append(frame)
        return resized

    @staticmethod
    def _instruction(obs: dict) -> str:
        """The episode instruction: the first ``annotation.`` / ``language.`` key."""
        key = next(k for k in obs if k.startswith("annotation.") or k.startswith("language."))
        return str(obs[key])

    def _save_png_frames(self, obs: dict) -> None:
        """step_count is 1 at reset and +1 per step; the PNG index is 0-based."""
        for key, frame in keep_canonical_video_obs(obs).items():
            if "video" not in key:
                continue
            camera_name = key.split(".", 1)[1]
            if self.video_modality_keys is not None and camera_name not in self.video_modality_keys:
                continue
            PILImage.fromarray(frame).save(self.frames_dir / f"step_{self.step_count - 1:04d}_{camera_name}.png")

    def _seed_env(self, episode_seed: int) -> None:
        """Process-global RNGs plus the robosuite env's own rng (robocasa kitchen / GR-1 tabletop keep one and
        their reset() takes no seed; the gym wrapper holds the robosuite env as ``.env``). LiberoEnv has no
        such rng."""
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        inner = self.env.unwrapped
        if hasattr(inner, "env"):
            inner.env.seed = episode_seed
            inner.env.rng = np.random.default_rng(episode_seed)

    def _finish_episode(self) -> None:
        """Rename the previous episode's mp4 (and frames dir) with its outcome, or remove it when unfinished."""
        self.video_recorder.stop()
        if self.file_path is None:
            return
        stem = f"{self.file_path.stem}-" + ("success" if self.is_success else "failure")
        # step_count is 1 at reset and +1 per step
        if self.step_count >= self.max_episode_steps or self.is_success:
            self.file_path.rename(self.output_dir / f"{stem}.mp4")
            if self.save_frames:
                self.frames_dir.rename(self.output_dir / "saved_frames" / stem)
        else:
            print(f"Skipping video recording for unfinished episode {self.step_count - 1} / {self.max_episode_steps}")
            self.file_path.unlink(missing_ok=True)
            if self.save_frames:
                shutil.rmtree(self.frames_dir)
        self.file_path = None
        self.frames_dir = None

    def reset(self, *, seed=None, options=None):
        assert seed is None, "episodes are seeded from base_seed and the episode id"
        episode_id = None
        if self._pending_cursor < len(self._pending_episode_ids):
            episode_id = self._pending_episode_ids[self._pending_cursor]
            self._pending_cursor += 1
            seed = self.base_seed * self.seed_stride + episode_id - 1
            self._seed_env(seed)
        obs, info = super().reset(seed=seed, options=options)

        self._finish_episode()
        self.is_success = False
        self.step_count = 1
        if episode_id is not None:
            self.file_path = self.output_dir / f"{self.name_prefix}-episode_{episode_id}.mp4"
            if self.save_frames:
                self.frames_dir = self.output_dir / "saved_frames" / self.file_path.stem
                self.frames_dir.mkdir(parents=True, exist_ok=True)
                (self.frames_dir / "instruction.txt").write_text(self._instruction(obs))
                self._save_png_frames(obs)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        self.step_count += 1
        self.is_success |= bool(info["success"])
        if self.file_path is not None:
            if not self.video_recorder.is_ready():
                self.video_recorder.start(self.file_path)
            if self.save_frames:
                self._save_png_frames(obs)
            video_keys = dedup_resolution_duplicate_video_keys([k for k in obs if "video" in k])
            assert video_keys, "No video frame found in the observation"
            frames = [obs[k] for k in video_keys]
            if len(frames) > 1:
                frames = self._resize_frames_to_common_height(frames)
            frame = np.ascontiguousarray(np.concatenate(frames, axis=1))
            self._overlay_text(frame, f"{self._instruction(obs)} ({int(info['success'])})")
            self.video_recorder.write_frame(frame)
        return obs, reward, done, truncated, info

    @staticmethod
    def _overlay_text(frame: np.ndarray, text: str) -> None:
        """Draw ``text`` on a dark band at the bottom of ``frame`` (in place), scaled to the frame width."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        padding = 5
        target_width = frame.shape[1] - 2 * padding
        scale = 1.0
        size = cv2.getTextSize(text, font, scale, thickness)[0]
        if size[0] > target_width:
            while size[0] > target_width and scale > 0.1:
                scale *= 0.9
                size = cv2.getTextSize(text, font, scale, thickness)[0]
        else:
            while size[0] < target_width and scale < 2.0:
                scale *= 1.1
                size = cv2.getTextSize(text, font, scale, thickness)[0]
            scale *= 0.9  # back off slightly so it fits
        x, y = padding, frame.shape[0] - 20
        cv2.rectangle(frame, (x - padding, y - size[1] - padding), (x + size[0] + padding, y + padding), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness)
