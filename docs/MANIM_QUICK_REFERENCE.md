# Manim Visualization Quick Reference

**Beautiful brain activity animations using Manim (3Blue1Brown's library)**

## Installation

```bash
# Basic Manim
pip install manim

# With LaTeX support (better text rendering)
pip install manim[all]

# Or install Thalia with visualization extras
pip install -e ".[visualization]"
```

## Quick Start

```python
from thalia.visualization import BrainActivityVisualization

# Create visualizer
viz = BrainActivityVisualization(
    checkpoint_path="training_runs/00_sensorimotor/checkpoints/stage_0_step_10000.thalia"
)

# Render architecture
viz.render_architecture("brain_architecture.mp4", quality="medium_quality")

# Render spike activity
viz.render_spikes("spike_activity.mp4", n_timesteps=100)

# Render learning (before/after)
viz.render_learning(
    output_path="learning.mp4",
    checkpoint_before="checkpoints/step_1000.thalia",
    checkpoint_after="checkpoints/step_10000.thalia"
)

# Render neurogenesis
viz.render_growth(
    output_path="growth.mp4",
    checkpoints=["step_1000.thalia", "step_2000.thalia", "step_3000.thalia"]
)
```

## Visualization Types

### 1. **Brain Architecture**

Shows the structure of brain regions and connections.

```python
viz.render_architecture(
    output_path="architecture.mp4",
    quality="high_quality"  # low, medium, high, production
)
```

**Features:**
- 3D brain regions as spheres
- Color-coded by function
- Rotating camera view
- Region labels

**Use for:**
- Overview slides
- Architecture diagrams
- Publications

---

### 2. **Spike Activity**

Animates spikes propagating through the network.

```python
viz.render_spikes(
    output_path="spikes.mp4",
    n_timesteps=100,
    quality="medium_quality"
)
```

**Features:**
- Neurons as dots
- Flashing spikes
- Activity propagation
- Color-coded by region

**Use for:**
- Demonstrating spike-based computation
- Showing temporal dynamics
- Explaining spiking networks

---

### 3. **Learning (Before/After)**

Shows synaptic weights changing during learning.

```python
viz.render_learning(
    output_path="learning.mp4",
    checkpoint_before="before.thalia",
    checkpoint_after="after.thalia",
    quality="medium_quality"
)
```

**Features:**
- Weak connections → strong connections
- STDP/BCM/Hebbian learning rules
- Weight visualization (line thickness)
- Learning equation overlay

**Use for:**
- Explaining plasticity
- Showing STDP in action
- Comparing before/after training

---

### 4. **Neurogenesis (Growth)**

Shows new neurons being added over time.

```python
viz.render_growth(
    output_path="growth.mp4",
    checkpoints=[
        "step_1000.thalia",
        "step_5000.thalia",
        "step_10000.thalia",
        "step_20000.thalia"
    ],
    quality="medium_quality"
)
```

**Features:**
- Network growing over time
- New neurons appearing
- Connections forming
- Neuron count updating

**Use for:**
- Demonstrating curriculum learning
- Showing developmental stages
- Explaining growth strategies

---

## Quality Settings

| Quality | Resolution | Framerate | Use Case |
|---------|-----------|-----------|----------|
| `low_quality` | 480p | 15fps | Quick preview |
| `medium_quality` | 720p | 30fps | Demos, presentations |
| `high_quality` | 1080p | 60fps | Professional videos |
| `production_quality` | 1440p | 60fps | Publications, final output |

**Tip:** Start with `low_quality` to iterate quickly, then render final version with `high_quality`.

---

## Advanced Usage

### Custom Scenes

Create your own Manim scenes:

```python
from thalia.visualization import BrainArchitectureScene
from manim import *

class CustomBrainScene(BrainArchitectureScene):
    def construct(self):
        # Your custom animation logic
        title = Text("My Custom Brain Visualization")
        self.play(Write(title))
        
        # Load checkpoint data
        regions = self._create_regions()
        
        # Custom animations
        for region in regions.values():
            self.play(FadeIn(region), run_time=0.5)
        
        self.wait(2)
```

### Extract Frames for Figures

```bash
# Extract specific frame (e.g., frame 100)
ffmpeg -i video.mp4 -vf "select=eq(n\,100)" -vsync 0 frame.png

# Extract last frame
ffmpeg -sseof -3 -i video.mp4 -update 1 -q:v 1 final_frame.png

# Extract every 10th frame
ffmpeg -i video.mp4 -vf "select=not(mod(n\,10))" frames_%04d.png
```

### Convert to GIF

```bash
# Create GIF from video (480px width)
ffmpeg -i video.mp4 -vf "scale=480:-1:flags=lanczos,fps=15" animation.gif

# Higher quality GIF
ffmpeg -i video.mp4 -vf "scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" output.gif
```

### Render with CLI

Use Manim's CLI directly for more control:

```bash
# Render specific scene
manim -pql src/thalia/visualization/manim_brain.py BrainArchitectureScene

# Options:
# -p: Preview after rendering
# -q: Quality (l=low, m=medium, h=high, k=4k)
# -s: Save last frame only (for figures)
# --format gif: Output as GIF
```

---

## Examples

### Full Example Script

See `examples/visualize_brain.py`:

```python
from thalia.visualization import BrainActivityVisualization

viz = BrainActivityVisualization("checkpoint.thalia")

# Render all visualizations
viz.render_architecture("arch.mp4")
viz.render_spikes("spikes.mp4", n_timesteps=100)
viz.render_learning("learning.mp4", 
                   checkpoint_before="before.thalia",
                   checkpoint_after="after.thalia")
viz.render_growth("growth.mp4", 
                 checkpoints=["step1.thalia", "step2.thalia", "step3.thalia"])

print("✅ All videos rendered!")
```

Run with:
```bash
python examples/visualize_brain.py
```

---

## Troubleshooting

### "manim not found"

```bash
pip install manim
```

### LaTeX errors

Either:
1. Install full Manim: `pip install manim[all]`
2. Or disable LaTeX in scenes (text rendering still works)

### Slow rendering

- Use `quality="low_quality"` for faster iteration
- Reduce `n_timesteps` for spike visualizations
- Use fewer checkpoints for growth visualization

### Video won't play

- Try VLC media player (handles all codecs)
- Or convert format: `ffmpeg -i output.mp4 -c:v libx264 output_converted.mp4`

---

## Tips & Best Practices

### For Presentations

```python
# Medium quality is perfect for slides
viz.render_spikes("demo.mp4", quality="medium_quality")
```

### For Publications

```python
# High quality + extract frame
viz.render_architecture("figure.mp4", quality="high_quality")
# Then: ffmpeg -sseof -1 -i figure.mp4 -update 1 figure.png
```

### For Social Media

```python
# Medium quality + convert to GIF
viz.render_learning("learning.mp4", quality="medium_quality")
# Then: ffmpeg -i learning.mp4 -vf scale=480:-1 learning.gif
```

### For YouTube/Demos

```python
# Production quality
viz.render_growth("demo.mp4", quality="production_quality")
```

---

## Comparison: Matplotlib vs Manim

| Feature | Matplotlib | Manim |
|---------|-----------|-------|
| **Purpose** | Real-time monitoring | Educational videos |
| **Speed** | Instant | Minutes to render |
| **Quality** | Good | Exceptional |
| **Animations** | Basic | Professional |
| **Use Case** | Training debugging | Papers, presentations |
| **Learning Curve** | Easy | Moderate |

**Recommendation:** Use matplotlib for monitoring during training, Manim for creating final videos/figures.

---

## Integration with Training

Monitor with matplotlib during training:
```python
from thalia.training import TrainingMonitor

monitor = TrainingMonitor("training_runs/00_sensorimotor")
monitor.show_metrics()  # Real-time matplotlib plots
```

Then create Manim video after training:
```python
from thalia.visualization import BrainActivityVisualization

viz = BrainActivityVisualization("training_runs/00_sensorimotor/checkpoints/final.thalia")
viz.render_architecture("final_brain.mp4", quality="high_quality")
```

---

## Further Resources

- **Manim Documentation**: https://docs.manim.community/
- **3Blue1Brown Videos**: https://www.3blue1brown.com/ (see source: https://github.com/3b1b/manim)
- **Manim Tutorial**: https://docs.manim.community/en/stable/tutorials/quickstart.html
- **Example Gallery**: https://docs.manim.community/en/stable/examples.html

---

**Last Updated**: December 9, 2025
