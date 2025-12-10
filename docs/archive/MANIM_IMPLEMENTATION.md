# Manim Visualization Module - Implementation Summary

**Date**: December 9, 2025  
**Status**: ✅ Complete and ready to use

## What Was Created

### Core Module: `src/thalia/visualization/manim_brain.py`

Four scene types for different visualization needs:

1. **BrainArchitectureScene** - 3D brain structure
   - Shows regions as colored spheres
   - Labels for each region
   - Rotating camera view
   - Use: Architecture diagrams, overview slides

2. **SpikeActivityScene** - Animated spike propagation
   - Neurons as dots
   - Spikes as flashes
   - Activity flows between regions
   - Use: Demonstrating spike-based computation

3. **LearningScene** - Before/after learning comparison
   - Synaptic weights as connections
   - Weak → strong connections
   - Learning rule equations
   - Use: Explaining STDP/BCM/Hebbian learning

4. **GrowthScene** - Neurogenesis over time
   - Network grows with new neurons
   - Connections form dynamically
   - Neuron count updates
   - Use: Showing curriculum learning stages

### High-Level API: `BrainActivityVisualization`

Simple interface for creating videos:

```python
from thalia.visualization import BrainActivityVisualization

viz = BrainActivityVisualization("checkpoint.thalia")
viz.render_architecture("brain.mp4")
viz.render_spikes("spikes.mp4", n_timesteps=100)
viz.render_learning("learning.mp4", checkpoint_before="...", checkpoint_after="...")
viz.render_growth("growth.mp4", checkpoints=[...])
```

### Documentation: `docs/MANIM_QUICK_REFERENCE.md`

Complete guide with:
- Installation instructions
- Quick start examples
- All scene types explained
- Quality settings guide
- Frame extraction (for figures)
- GIF conversion
- CLI usage
- Troubleshooting
- Best practices
- Matplotlib vs Manim comparison

### Example: `examples/visualize_brain.py`

Runnable script demonstrating all visualization types.

## Installation

```bash
# Basic Manim
pip install manim

# Or with Thalia
pip install -e ".[visualization]"
```

## Key Design Decisions

### Why Manim?

- **Professional quality**: 3Blue1Brown-level animations
- **Educational**: Perfect for papers, presentations, demos
- **Flexible**: Can create custom scenes
- **Ecosystem**: Large community, good docs

### Why Not Replace Matplotlib?

Kept both because they serve different purposes:

| Use Case | Tool | Reason |
|----------|------|--------|
| Training monitoring | Matplotlib | Instant, real-time, simple |
| Demo videos | Manim | Beautiful, professional |
| Publications | Manim | High-quality figures |
| Debugging | Matplotlib | Fast iteration |
| Social media | Manim | Eye-catching animations |

### Architecture

- **Graceful degradation**: Works even if Manim not installed
- **Checkpoint integration**: Loads directly from `.thalia` files
- **Quality levels**: Low (fast preview) → Production (final output)
- **Multiple formats**: Video (MP4) + frame extraction (PNG) + GIF

## What Changed

### New Files
```
src/thalia/visualization/__init__.py
src/thalia/visualization/manim_brain.py
examples/visualize_brain.py
docs/MANIM_QUICK_REFERENCE.md
```

### Modified Files
```
pyproject.toml                    # Added visualization extras
src/thalia/__init__.py           # Export BrainActivityVisualization
docs/README.md                   # Added visualization link
```

## Usage Examples

### Quick Demo
```python
from thalia.visualization import BrainActivityVisualization

viz = BrainActivityVisualization("training_runs/00_sensorimotor/checkpoints/final.thalia")
viz.render_architecture("brain.mp4", quality="medium_quality")
```

### For Publications
```python
# High quality video
viz.render_learning("figure_3.mp4", 
                   checkpoint_before="before.thalia",
                   checkpoint_after="after.thalia",
                   quality="high_quality")

# Extract last frame as PNG
# ffmpeg -sseof -1 -i figure_3.mp4 -update 1 figure_3.png
```

### For Social Media
```python
# Medium quality + convert to GIF
viz.render_spikes("demo.mp4", n_timesteps=100, quality="medium_quality")
# ffmpeg -i demo.mp4 -vf scale=480:-1 demo.gif
```

### Custom Scene
```python
from thalia.visualization import SpikeActivityScene
from manim import *

class MySpikeScene(SpikeActivityScene):
    def construct(self):
        # Your custom animation
        title = Text("My Custom Visualization")
        self.play(Write(title))
        # ... more custom code
```

## Testing

The module handles missing Manim gracefully:

```python
from thalia.visualization import MANIM_AVAILABLE

if MANIM_AVAILABLE:
    viz = BrainActivityVisualization("checkpoint.thalia")
    viz.render_architecture("brain.mp4")
else:
    print("Install manim: pip install manim")
```

## Next Steps (Optional Enhancements)

Future additions could include:

1. **More scene types**:
   - Reward prediction (dopamine)
   - Memory replay (hippocampus)
   - Attention mechanisms (prefrontal)
   - Multi-region coordination

2. **Interactive features**:
   - Choose which regions to show
   - Highlight specific connections
   - Time slider for checkpoint progression

3. **Data-driven animations**:
   - Load actual spike trains from checkpoints
   - Show real weight matrices
   - Visualize actual learning curves

4. **Integration with training**:
   - Auto-generate videos after each stage
   - Progress videos (checkpoint every N steps)
   - Side-by-side comparisons

5. **Templates**:
   - Paper figure templates
   - Presentation slide templates
   - Social media templates (square, vertical)

## Benefits

### For You
- ✅ High-quality demo videos
- ✅ Publication-ready figures
- ✅ Explainer content for papers
- ✅ Social media content
- ✅ Presentation slides

### For the Project
- ✅ Professional appearance
- ✅ Educational value
- ✅ Clear demonstrations
- ✅ Shareable content
- ✅ Research communication

## Comparison Summary

| Feature | Matplotlib | Manim |
|---------|-----------|-------|
| **Speed** | Instant | Minutes |
| **Quality** | Good | Exceptional |
| **Use Case** | Monitoring | Videos/Figures |
| **Learning Curve** | Easy | Moderate |
| **Real-time** | Yes | No |
| **Animations** | Basic | Professional |

**Both are valuable** - use the right tool for the job!

---

## Questions?

- **Installation help**: See `docs/MANIM_QUICK_REFERENCE.md`
- **Examples**: Run `python examples/visualize_brain.py`
- **Custom scenes**: See Manim docs at https://docs.manim.community/
- **Troubleshooting**: Check quick reference guide

**The module is ready to use!** Install Manim and start creating beautiful brain visualizations. 🎬🧠
