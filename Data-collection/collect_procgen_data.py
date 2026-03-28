import os
import numpy as np
import h5py
from procgen import ProcgenEnv
from tqdm import tqdm
from PIL import Image
import time


#Collection
#============================================================================

def collect_and_save(game, num_steps, num_levels, start_level, output_path):
    """Collect transitions using random policy and save to HDF5."""
    # Use procgen's native API directly — no gym registration needed
    env = ProcgenEnv(
        num_envs=1,
        env_name=game,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode="easy",
    )
    num_actions = env.action_space.n

    print(f"\n  Game:       {game}")
    print(f"  Levels:     {start_level} - {start_level + num_levels - 1}")
    print(f"  Steps:      {num_steps:,}")
    print(f"  Actions:    {num_actions}")
    print(f"  Output:     {output_path}")

    frames = []
    actions = []
    next_frames = []
    rewards = []
    dones = []

    obs = env.reset()["rgb"][0]  # [64, 64, 3]
    ep_count = 0
    ep_reward = 0
    ep_rewards = []

    for _ in tqdm(range(num_steps), desc=f"{game}", ncols=80):
        action = np.array([np.random.randint(num_actions)])
        obs_dict, reward, done, info = env.step(action)

        next_obs = obs_dict["rgb"][0]      # [64, 64, 3]
        reward = reward[0]                 # scalar
        done = done[0]                     # bool

        frames.append(obs)
        actions.append(action[0])
        next_frames.append(next_obs)
        rewards.append(reward)
        dones.append(done)
        ep_reward += reward

        if done:
            ep_count += 1
            ep_rewards.append(ep_reward)
            ep_reward = 0
            # procgen auto-resets, next_obs is already the new episode's first frame

        obs = next_obs

    env.close()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("frames", data=np.array(frames, dtype=np.uint8),
                         compression="gzip", compression_opts=4,
                         chunks=(100, 64, 64, 3))
        f.create_dataset("actions", data=np.array(actions, dtype=np.int32),
                         compression="gzip")
        f.create_dataset("next_frames", data=np.array(next_frames, dtype=np.uint8),
                         compression="gzip", compression_opts=4,
                         chunks=(100, 64, 64, 3))
        f.create_dataset("rewards", data=np.array(rewards, dtype=np.float32),
                         compression="gzip")
        f.create_dataset("dones", data=np.array(dones, dtype=bool),
                         compression="gzip")
        f.attrs["num_episodes"] = ep_count
        f.attrs["action_space_size"] = num_actions
        f.attrs["mean_episode_reward"] = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        f.attrs["game"] = game
        f.attrs["num_levels"] = num_levels
        f.attrs["start_level"] = start_level

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    mean_reward = np.mean(ep_rewards) if ep_rewards else 0
    print(f"  Saved: {size_mb:.1f} MB | {ep_count} episodes | mean reward: {mean_reward:.2f}")

    # Free memory
    del frames, actions, next_frames, rewards, dones


def save_samples(h5_path, sample_dir, num_samples=10):
    """Save sample frame pairs as images for quick visual check."""
    os.makedirs(sample_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        n = f["frames"].shape[0]
        indices = np.linspace(0, n - 1, num_samples, dtype=int)

        for i, idx in enumerate(indices):
            frame = Image.fromarray(f["frames"][idx])
            next_frame = Image.fromarray(f["next_frames"][idx])
            action = f["actions"][idx]

            # Side by side, scaled up 4x
            combined = Image.new("RGB", (64 * 2 + 10, 64), color=(30, 30, 30))
            combined.paste(frame, (0, 0))
            combined.paste(next_frame, (74, 0))
            combined = combined.resize((combined.width * 4, combined.height * 4),
                                       Image.NEAREST)
            combined.save(os.path.join(sample_dir, f"{i:03d}_action{action}.png"))

    print(f"  Samples saved to {sample_dir}/")


#Main
#============================================================================

def main():
    output_dir = "procgen_data"
    games = ["coinrun", "starpilot"]
    train_steps = 100_000
    test_steps = 20_000

    print("=" * 60)
    print("  PROCGEN DATA COLLECTION")
    print("=" * 60)
    print(f"  Games:       {games}")
    print(f"  Train steps: {train_steps:,}")
    print(f"  Test steps:  {test_steps:,}")
    print(f"  Output:      {output_dir}/")
    print("=" * 60)

    start = time.time()

    for game in games:
        print(f"\n{'='*60}")
        print(f"  {game.upper()}")
        print(f"{'='*60}")

        # Train data (levels 0-99)
        print("\n[Train]")
        train_path = os.path.join(output_dir, f"{game}_train.h5")
        collect_and_save(game, train_steps, num_levels=100,
                         start_level=0, output_path=train_path)
        save_samples(train_path, os.path.join(output_dir, "samples", f"{game}_train"))

        # Test data (levels 100-199)
        print("\n[Test]")
        test_path = os.path.join(output_dir, f"{game}_test.h5")
        collect_and_save(game, test_steps, num_levels=100,
                         start_level=100, output_path=test_path)
        save_samples(test_path, os.path.join(output_dir, "samples", f"{game}_test"))

    elapsed = time.time() - start

    # Summary
    print(f"\n{'='*60}")
    print(f"  DONE — {elapsed/60:.1f} minutes")
    print(f"{'='*60}\n")

    total_size = 0
    for root, dirs, files in os.walk(output_dir):
        for fname in sorted(files):
            if fname.endswith(".h5"):
                fpath = os.path.join(root, fname)
                size = os.path.getsize(fpath) / (1024 ** 2)
                total_size += size
                print(f"  {fname:30s} {size:8.1f} MB")

    print(f"\n  {'TOTAL':30s} {total_size:8.1f} MB")


if __name__ == "__main__":
    main()