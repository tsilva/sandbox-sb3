from __future__ import annotations

import argparse
import os
import time
from collections import deque

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pygame
from stable_baselines3 import PPO

from mario_ppo.env import EnvConfig, assert_rom_imported, make_mario_env


def stacked_obs(frames: deque[np.ndarray]) -> np.ndarray:
    # Model was trained with VecFrameStack + VecTransposeImage: (n_env, 4, 84, 84).
    return np.stack([frame[..., 0] for frame in frames], axis=0)[None, ...]


class PygameViewer:
    def __init__(self, frame_shape: tuple[int, int, int], scale: int, fps: float):
        if scale < 1:
            raise ValueError("--scale must be >= 1")
        height, width, _channels = frame_shape
        self.size = (width * scale, height * scale)
        self.fps = fps
        pygame.init()
        pygame.display.set_caption("Mario PPO")
        self.screen = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()

    def show(self, frame: np.ndarray) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surface = pygame.transform.scale(surface, self.size)
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        if self.fps > 0:
            self.clock.tick(self.fps)
        return True

    def close(self) -> None:
        pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show a PPO checkpoint playing Mario in a GUI window")
    parser.add_argument("--model", default="runs/smoke_doc/final_model.zip")
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assert_rom_imported()
    model = PPO.load(args.model)
    config = EnvConfig(state=args.state, max_episode_steps=args.max_steps)
    env = make_mario_env(config=config, seed=args.seed)

    obs, _ = env.reset(seed=args.seed)
    first_frame = env.render()
    viewer = PygameViewer(first_frame.shape, scale=args.scale, fps=args.fps)

    try:
        if not viewer.show(first_frame):
            return
        for episode in range(args.episodes):
            if episode > 0:
                obs, _ = env.reset(seed=args.seed + episode)
                frame = env.render()
                if not viewer.show(frame):
                    break
            frames: deque[np.ndarray] = deque([obs] * 4, maxlen=4)
            for _step in range(args.max_steps):
                action, _ = model.predict(stacked_obs(frames), deterministic=True)
                obs, _reward, terminated, truncated, _info = env.step(int(action[0]))
                frames.append(obs)
                frame = env.render()
                if not viewer.show(frame):
                    return
                if terminated or truncated:
                    time.sleep(0.5)
                    break
    finally:
        viewer.close()
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
