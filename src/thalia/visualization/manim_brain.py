"""
Manim Brain Visualization - Beautiful animations of brain activity.

This module uses Manim (3Blue1Brown's animation library) to create
educational visualizations of Thalia's brain activity.

Use cases:
- Demo videos showing learning in action
- Publication-quality figures
- Explanatory content for papers/presentations
- Showing spike propagation through network

Installation:
    pip install manim

    # Or with LaTeX support for better text rendering:
    pip install manim[all]

Usage:
======

    from thalia.visualization.manim_brain import BrainActivityVisualization
    
    # Create animation from checkpoint
    viz = BrainActivityVisualization(
        checkpoint_path="training_runs/00_sensorimotor/checkpoints/stage_0_step_10000.thalia"
    )
    
    # Render video
    viz.render("brain_activity.mp4", quality="medium_quality")
    
    # Or render just one frame (for figures)
    viz.render_frame("brain_snapshot.png")

Scene Types:
============

1. BrainArchitectureScene - Show brain structure
2. SpikeActivityScene - Animate spikes propagating
3. LearningScene - Show weights changing over time
4. GrowthScene - Show neurogenesis happening

Author: Thalia Project
Date: December 9, 2025
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

try:
    from manim import *
    MANIM_AVAILABLE = True
except ImportError:
    MANIM_AVAILABLE = False
    # Define dummy classes so module can still be imported
    class Scene:
        pass
    class ThreeDScene:
        pass
    class Mobject:
        pass


class BrainArchitectureScene(ThreeDScene):
    """
    Visualize the brain's architecture - regions and connections.
    
    Shows:
    - Brain regions as 3D objects
    - Connections between regions
    - Region labels
    - Can rotate camera around brain
    """
    
    def __init__(self, checkpoint_path: str, **kwargs):
        """
        Initialize scene.
        
        Args:
            checkpoint_path: Path to .thalia checkpoint file
        """
        if not MANIM_AVAILABLE:
            raise ImportError("manim is required. Install with: pip install manim")
        
        super().__init__(**kwargs)
        self.checkpoint_path = Path(checkpoint_path)
        self.brain_data = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load brain structure from checkpoint."""
        from thalia.io.checkpoint import BrainCheckpoint
        
        info = BrainCheckpoint.info(self.checkpoint_path)
        return info
    
    def construct(self):
        """Build the animation."""
        # Title
        title = Text("Thalia's Brain Architecture", font_size=48)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.wait()
        
        # Create brain regions as 3D objects
        regions = self._create_regions()
        
        # Show regions appearing one by one
        for region_name, region_obj in regions.items():
            label = Text(region_name, font_size=24)
            label.next_to(region_obj, UP)
            self.add_fixed_in_frame_mobjects(label)
            
            self.play(
                Create(region_obj),
                Write(label),
                run_time=0.5
            )
        
        self.wait()
        
        # Rotate camera around brain
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(10)
        self.stop_ambient_camera_rotation()
        
        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
    
    def _create_regions(self) -> Dict[str, Mobject]:
        """Create 3D objects for each brain region."""
        regions = {}
        
        # Define region positions (simplified brain layout)
        region_positions = {
            'visual_cortex': np.array([-2, -1, 0]),
            'motor_cortex': np.array([2, 1, 0]),
            'hippocampus': np.array([0, -0.5, -1]),
            'prefrontal': np.array([0, 2, 0.5]),
            'striatum': np.array([0.5, 0, -0.5]),
            'cerebellum': np.array([0, -2, -1]),
        }
        
        colors = {
            'visual_cortex': BLUE,
            'motor_cortex': RED,
            'hippocampus': GREEN,
            'prefrontal': PURPLE,
            'striatum': ORANGE,
            'cerebellum': YELLOW,
        }
        
        for region_name, position in region_positions.items():
            # Create sphere for each region
            sphere = Sphere(radius=0.4, resolution=(10, 10))
            sphere.set_color(colors.get(region_name, WHITE))
            sphere.set_opacity(0.7)
            sphere.move_to(position)
            regions[region_name] = sphere
            self.add(sphere)
        
        return regions


class SpikeActivityScene(Scene):
    """
    Animate spikes propagating through the network.
    
    Shows:
    - Neurons as dots
    - Spikes as flashes/pulses
    - Activity propagating between regions
    - Color-coded by region
    """
    
    def __init__(self, checkpoint_path: str, n_timesteps: int = 100, **kwargs):
        """
        Initialize scene.
        
        Args:
            checkpoint_path: Path to .thalia checkpoint file
            n_timesteps: Number of timesteps to simulate
        """
        if not MANIM_AVAILABLE:
            raise ImportError("manim is required. Install with: pip install manim")
        
        super().__init__(**kwargs)
        self.checkpoint_path = Path(checkpoint_path)
        self.n_timesteps = n_timesteps
        self.brain_data = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load brain state from checkpoint."""
        from thalia.io.checkpoint import BrainCheckpoint
        
        info = BrainCheckpoint.info(self.checkpoint_path)
        return info
    
    def construct(self):
        """Build the animation."""
        # Title
        title = Text("Spike Activity in Thalia's Brain", font_size=40)
        title.to_edge(UP)
        self.add(title)
        
        # Create simplified network visualization
        # Show a few neurons from each region
        neurons_per_region = 20
        regions = ['Visual', 'Motor', 'Hippocampus']
        colors = [BLUE, RED, GREEN]
        
        neuron_groups = []
        for i, (region, color) in enumerate(zip(regions, colors)):
            # Create grid of dots for neurons
            neurons = VGroup()
            for j in range(neurons_per_region):
                x = -4 + i * 4
                y = (j - neurons_per_region/2) * 0.3
                neuron = Dot(point=[x, y, 0], color=color, radius=0.08)
                neurons.add(neuron)
            
            neuron_groups.append(neurons)
            
            # Add region label
            label = Text(region, font_size=24, color=color)
            label.next_to(neurons, DOWN)
            self.add(label)
        
        # Add all neurons
        for group in neuron_groups:
            self.add(group)
        
        self.wait()
        
        # Simulate spike activity
        # Random spikes propagating left to right
        for t in range(50):
            # Pick random neurons to spike
            for group in neuron_groups:
                spike_indices = np.random.choice(
                    len(group),
                    size=np.random.randint(1, 5),
                    replace=False
                )
                
                animations = []
                for idx in spike_indices:
                    neuron = group[idx]
                    # Flash animation
                    animations.append(
                        neuron.animate.scale(2).set_opacity(1)
                    )
                
                if animations:
                    self.play(*animations, run_time=0.1)
                    # Return to normal
                    self.play(
                        *[neuron.animate.scale(0.5).set_opacity(0.7) 
                          for idx in spike_indices for neuron in [group[idx]]],
                        run_time=0.1
                    )
        
        self.wait()
        
        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class LearningScene(Scene):
    """
    Visualize learning - weights changing over time.
    
    Shows:
    - Synaptic weights as connections
    - Weights strengthening (thickening lines)
    - STDP/BCM learning rules in action
    - Before/after comparison
    """
    
    def __init__(self, checkpoint_before: str, checkpoint_after: str, **kwargs):
        """
        Initialize scene.
        
        Args:
            checkpoint_before: Path to checkpoint before learning
            checkpoint_after: Path to checkpoint after learning
        """
        if not MANIM_AVAILABLE:
            raise ImportError("manim is required. Install with: pip install manim")
        
        super().__init__(**kwargs)
        self.checkpoint_before = Path(checkpoint_before)
        self.checkpoint_after = Path(checkpoint_after)
    
    def construct(self):
        """Build the animation."""
        # Title
        title = Text("Synaptic Learning in Action", font_size=40)
        title.to_edge(UP)
        self.add(title)
        
        # Create simple 2-layer network
        layer1 = VGroup(*[Dot(point=[-3, i-2, 0], radius=0.15, color=BLUE) 
                         for i in range(5)])
        layer2 = VGroup(*[Dot(point=[3, i-2, 0], radius=0.15, color=GREEN) 
                         for i in range(5)])
        
        self.play(Create(layer1), Create(layer2))
        
        # Show weak connections initially
        connections_before = VGroup()
        for n1 in layer1:
            for n2 in layer2:
                line = Line(n1.get_center(), n2.get_center(), 
                           stroke_width=0.5, stroke_opacity=0.3, color=GRAY)
                connections_before.add(line)
        
        self.play(Create(connections_before))
        
        before_label = Text("Before Learning", font_size=30, color=RED)
        before_label.to_edge(DOWN)
        self.play(Write(before_label))
        self.wait(2)
        
        # Transform to strong connections after learning
        connections_after = VGroup()
        for n1 in layer1:
            for n2 in layer2:
                # Some connections get stronger (STDP/BCM)
                strength = np.random.random()
                if strength > 0.5:  # Potentiated
                    line = Line(n1.get_center(), n2.get_center(),
                               stroke_width=3, stroke_opacity=0.9, color=YELLOW)
                else:  # Depressed
                    line = Line(n1.get_center(), n2.get_center(),
                               stroke_width=0.2, stroke_opacity=0.1, color=GRAY)
                connections_after.add(line)
        
        after_label = Text("After Learning", font_size=30, color=GREEN)
        after_label.to_edge(DOWN)
        
        self.play(
            Transform(connections_before, connections_after),
            Transform(before_label, after_label),
            run_time=3
        )
        
        self.wait(2)
        
        # Show learning rule equation
        equation = MathTex(
            r"\Delta w = \eta \cdot x_{pre} \cdot x_{post}",
            font_size=36
        )
        equation.to_edge(UP).shift(DOWN)
        
        rule_label = Text("Hebbian Learning Rule", font_size=24)
        rule_label.next_to(equation, DOWN)
        
        self.play(Write(equation), Write(rule_label))
        self.wait(3)
        
        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class GrowthScene(Scene):
    """
    Visualize neurogenesis - new neurons being added.
    
    Shows:
    - Existing network
    - New neurons appearing
    - Connections forming to new neurons
    - Network growing over time
    """
    
    def __init__(self, checkpoints: List[str], **kwargs):
        """
        Initialize scene.
        
        Args:
            checkpoints: List of checkpoint paths showing growth progression
        """
        if not MANIM_AVAILABLE:
            raise ImportError("manim is required. Install with: pip install manim")
        
        super().__init__(**kwargs)
        self.checkpoints = [Path(cp) for cp in checkpoints]
    
    def construct(self):
        """Build the animation."""
        # Title
        title = Text("Neurogenesis: Brain Growth", font_size=40)
        title.to_edge(UP)
        self.add(title)
        
        # Start with small network
        neurons = VGroup()
        initial_count = 10
        
        for i in range(initial_count):
            angle = i * (2 * PI / initial_count)
            pos = np.array([np.cos(angle) * 2, np.sin(angle) * 2, 0])
            neuron = Dot(point=pos, radius=0.12, color=BLUE)
            neurons.add(neuron)
        
        self.play(Create(neurons))
        
        # Show initial neuron count
        count_label = Text(f"Neurons: {initial_count}", font_size=30)
        count_label.to_edge(DOWN)
        self.play(Write(count_label))
        
        self.wait()
        
        # Add neurons progressively
        for growth_step in range(5):
            # Add 3 new neurons
            new_neurons = VGroup()
            for j in range(3):
                # Random position in circle
                angle = np.random.random() * 2 * PI
                radius = 2 + np.random.random() * 0.5
                pos = np.array([np.cos(angle) * radius, np.sin(angle) * radius, 0])
                neuron = Dot(point=pos, radius=0.12, color=GREEN)
                new_neurons.add(neuron)
                neurons.add(neuron)
            
            # Animate growth
            self.play(
                *[GrowFromCenter(n) for n in new_neurons],
                run_time=0.5
            )
            
            # Connect to nearby neurons
            for new_neuron in new_neurons:
                # Find closest existing neuron
                distances = [np.linalg.norm(
                    new_neuron.get_center() - n.get_center()
                ) for n in neurons if n not in new_neurons]
                if distances:
                    closest_idx = np.argmin(distances)
                    closest_neuron = [n for n in neurons if n not in new_neurons][closest_idx]
                    
                    connection = Line(
                        new_neuron.get_center(),
                        closest_neuron.get_center(),
                        stroke_width=2,
                        color=YELLOW
                    )
                    self.play(Create(connection), run_time=0.3)
            
            # Update count
            new_count = initial_count + (growth_step + 1) * 3
            new_label = Text(f"Neurons: {new_count}", font_size=30)
            new_label.to_edge(DOWN)
            self.play(Transform(count_label, new_label))
            
            # Change new neurons from green to blue (integrated)
            self.play(*[n.animate.set_color(BLUE) for n in new_neurons], run_time=0.5)
            
            self.wait(0.5)
        
        self.wait(2)
        
        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class BrainActivityVisualization:
    """
    Main interface for creating brain visualizations.
    
    Usage:
        viz = BrainActivityVisualization(checkpoint_path="...")
        viz.render_architecture("output.mp4")
        viz.render_spikes("spikes.mp4")
        viz.render_learning("learning.mp4", checkpoint_before="...", checkpoint_after="...")
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize visualization.
        
        Args:
            checkpoint_path: Default checkpoint to use
        """
        if not MANIM_AVAILABLE:
            raise ImportError(
                "manim is required for brain visualizations.\n"
                "Install with: pip install manim\n"
                "For full features: pip install manim[all]"
            )
        
        self.checkpoint_path = checkpoint_path
    
    def render_architecture(
        self,
        output_path: str,
        checkpoint_path: Optional[str] = None,
        quality: str = "medium_quality"
    ) -> None:
        """
        Render brain architecture visualization.
        
        Args:
            output_path: Output video file path
            checkpoint_path: Checkpoint to visualize (uses default if None)
            quality: Manim quality setting ('low_quality', 'medium_quality', 'high_quality')
        """
        cp = checkpoint_path or self.checkpoint_path
        if not cp:
            raise ValueError("No checkpoint path specified")
        
        scene = BrainArchitectureScene(cp)
        scene.render(output_path, quality=quality)
        print(f"✅ Architecture video saved: {output_path}")
    
    def render_spikes(
        self,
        output_path: str,
        checkpoint_path: Optional[str] = None,
        n_timesteps: int = 100,
        quality: str = "medium_quality"
    ) -> None:
        """
        Render spike activity visualization.
        
        Args:
            output_path: Output video file path
            checkpoint_path: Checkpoint to visualize
            n_timesteps: Number of simulation timesteps
            quality: Manim quality setting
        """
        cp = checkpoint_path or self.checkpoint_path
        if not cp:
            raise ValueError("No checkpoint path specified")
        
        scene = SpikeActivityScene(cp, n_timesteps=n_timesteps)
        scene.render(output_path, quality=quality)
        print(f"✅ Spike activity video saved: {output_path}")
    
    def render_learning(
        self,
        output_path: str,
        checkpoint_before: str,
        checkpoint_after: str,
        quality: str = "medium_quality"
    ) -> None:
        """
        Render learning visualization (before/after).
        
        Args:
            output_path: Output video file path
            checkpoint_before: Checkpoint before learning
            checkpoint_after: Checkpoint after learning
            quality: Manim quality setting
        """
        scene = LearningScene(checkpoint_before, checkpoint_after)
        scene.render(output_path, quality=quality)
        print(f"✅ Learning video saved: {output_path}")
    
    def render_growth(
        self,
        output_path: str,
        checkpoints: List[str],
        quality: str = "medium_quality"
    ) -> None:
        """
        Render neurogenesis visualization.
        
        Args:
            output_path: Output video file path
            checkpoints: List of checkpoints showing growth progression
            quality: Manim quality setting
        """
        scene = GrowthScene(checkpoints)
        scene.render(output_path, quality=quality)
        print(f"✅ Growth video saved: {output_path}")
    
    def render_frame(
        self,
        output_path: str,
        scene_type: str = "architecture",
        **kwargs
    ) -> None:
        """
        Render single frame (for publication figures).
        
        Args:
            output_path: Output PNG file path
            scene_type: Type of scene ('architecture', 'spikes', 'learning', 'growth')
            **kwargs: Scene-specific arguments
        """
        # Render last frame only
        quality = "high_quality"
        if scene_type == "architecture":
            self.render_architecture(output_path.replace('.png', '.mp4'), quality=quality)
        elif scene_type == "spikes":
            self.render_spikes(output_path.replace('.png', '.mp4'), quality=quality)
        elif scene_type == "learning":
            self.render_learning(output_path.replace('.png', '.mp4'), **kwargs, quality=quality)
        elif scene_type == "growth":
            self.render_growth(output_path.replace('.png', '.mp4'), **kwargs, quality=quality)
        
        # Extract last frame (would need ffmpeg or PIL)
        print(f"📷 To extract frame: ffmpeg -sseof -3 -i {output_path.replace('.png', '.mp4')} -update 1 -q:v 1 {output_path}")
