"""
Example: Creating Brain Visualizations with Manim

This script demonstrates how to create beautiful animations of brain activity
using Manim (3Blue1Brown's animation library).

Prerequisites:
    pip install manim
    
    # Or with LaTeX support for better text:
    pip install manim[all]

Usage:
    python examples/visualize_brain.py

Output:
    - brain_architecture.mp4: Shows brain structure
    - spike_activity.mp4: Shows spikes propagating
    - learning.mp4: Shows synaptic plasticity
    - growth.mp4: Shows neurogenesis

Author: Thalia Project
Date: December 9, 2025
"""

from pathlib import Path
import warnings

# Suppress pydub ffmpeg warning (not needed for visual-only Manim animations)
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")

from thalia.visualization import BrainActivityVisualization, MANIM_AVAILABLE


def main():
    """Create brain visualizations."""
    
    if not MANIM_AVAILABLE:
        print("❌ Manim not installed.")
        print("Install with: pip install manim")
        return
    
    # Find training checkpoints
    training_dir = Path("training_runs")
    
    # Example 1: Brain Architecture
    # =============================
    print("\n📹 Rendering brain architecture...")
    checkpoint = training_dir / "00_sensorimotor" / "checkpoints" / "stage_0_step_10000.thalia"
    
    if checkpoint.exists():
        viz = BrainActivityVisualization(checkpoint_path=str(checkpoint))
        viz.render_architecture(
            output_path="brain_architecture.mp4",
            quality="medium_quality"  # 'low_quality', 'medium_quality', 'high_quality', 'production_quality'
        )
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint}")
        print("Run training first to generate checkpoints")
    
    # Example 2: Spike Activity
    # =========================
    print("\n📹 Rendering spike activity...")
    if checkpoint.exists():
        viz.render_spikes(
            output_path="spike_activity.mp4",
            n_timesteps=100,
            quality="medium_quality"
        )
    
    # Example 3: Learning (Before/After)
    # ===================================
    print("\n📹 Rendering learning comparison...")
    checkpoint_before = training_dir / "00_sensorimotor" / "checkpoints" / "stage_0_step_1000.thalia"
    checkpoint_after = training_dir / "00_sensorimotor" / "checkpoints" / "stage_0_step_10000.thalia"
    
    if checkpoint_before.exists() and checkpoint_after.exists():
        viz.render_learning(
            output_path="learning.mp4",
            checkpoint_before=str(checkpoint_before),
            checkpoint_after=str(checkpoint_after),
            quality="medium_quality"
        )
    else:
        print("⚠️  Need two checkpoints for learning comparison")
    
    # Example 4: Growth Over Time
    # ============================
    print("\n📹 Rendering neurogenesis...")
    checkpoints_dir = training_dir / "00_sensorimotor" / "checkpoints"
    
    if checkpoints_dir.exists():
        # Get multiple checkpoints showing progression
        checkpoints = sorted(checkpoints_dir.glob("*.thalia"))[:5]
        
        if len(checkpoints) >= 2:
            viz.render_growth(
                output_path="growth.mp4",
                checkpoints=[str(cp) for cp in checkpoints],
                quality="medium_quality"
            )
        else:
            print("⚠️  Need multiple checkpoints for growth visualization")
    
    print("\n✅ Done! Videos saved:")
    print("   - brain_architecture.mp4")
    print("   - spike_activity.mp4")
    print("   - learning.mp4")
    print("   - growth.mp4")
    
    print("\n💡 Tips:")
    print("   - Use quality='high_quality' for better visuals (slower)")
    print("   - Use quality='production_quality' for publication (very slow)")
    print("   - Extract frames: ffmpeg -i video.mp4 -vf 'select=eq(n\\,100)' -vsync 0 frame.png")
    print("   - Convert to GIF: ffmpeg -i video.mp4 -vf scale=480:-1 animation.gif")


if __name__ == "__main__":
    main()
