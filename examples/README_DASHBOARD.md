# Thalia Curriculum Training Dashboard

Real-time monitoring dashboard for Thalia's curriculum training system built with Streamlit.

## Features

### 📊 Real-Time Metrics
- **Firing Rate**: Current average firing rate across regions with delta
- **Capacity**: System capacity utilization percentage
- **Performance**: Task performance accuracy
- **Loss**: Training loss with trend indicators
- Historical charts for all metrics

### 🌱 Growth Monitoring
- **Neuron Count Over Time**: Track neuron additions per region
- **Capacity by Region**: Bar chart showing per-region capacity
- **Growth Events Log**: Detailed table of growth events with reasons
- Visual timeline of system expansion

### 💤 Consolidation Tracking
- **Consolidation Timeline**: When and how often consolidation occurs
- **Patterns Replayed**: Number of patterns replayed per event
- **Event Statistics**: Total events, average patterns, duration
- Recent consolidation events table

### ✅ Milestone Checklist
- **Completion Progress**: Overall milestone completion percentage
- **Detailed Status**: Per-milestone pass/fail indicators
- **Stage-Specific Criteria**: Automatically loads stage milestones
- Visual progress bar

### ⚠️ Health Warnings
- **System Health**: Automatic health check warnings
- **Severity Classification**: Color-coded warnings (info/warning/critical)
- **Real-Time Alerts**: Immediate notification of issues
- Prominent display when warnings exist

### ⚙️ Configuration
- **Checkpoint Directory**: Configurable checkpoint path
- **Auto-Refresh**: Toggle automatic updates (1-30 second intervals)
- **Display Options**: Show/hide detailed sections
- **Manual Refresh**: On-demand refresh button

## Installation

1. **Install Dashboard Dependencies**:
   ```bash
   pip install -r requirements-dashboard.txt
   ```

2. **Verify Streamlit Installation**:
   ```bash
   streamlit --version
   ```

## Usage

### Basic Usage

Run the dashboard with default checkpoint directory:

```bash
streamlit run examples/curriculum_dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

### Custom Checkpoint Directory

Specify a custom checkpoint directory via the sidebar or command line:

```bash
streamlit run examples/curriculum_dashboard.py
```

Then enter your checkpoint path in the sidebar: `checkpoints/my_training`

### Configuration Options

**Sidebar Settings**:
- **Checkpoint Directory**: Path to checkpoint directory (default: `checkpoints/curriculum`)
- **Auto-refresh**: Enable automatic dashboard updates
- **Refresh Rate**: Update interval in seconds (1-30)
- **Show Detailed Metrics**: Toggle historical metric charts
- **Show Growth Details**: Toggle growth event details table
- **Show Consolidation Details**: Toggle consolidation event log

## Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│ 🧠 Thalia Curriculum Training Monitor                       │
├─────────────────────────────────────────────────────────────┤
│ Stage Info │ Week Progress │ Training Step │ Last Update   │
│ [========= Stage Progress Bar ==========]                   │
├─────────────────────────────────────────────────────────────┤
│ ⚠️ Health Warnings (if any)                                 │
├─────────────────────────────────────────────────────────────┤
│ 📊 Current Metrics                                          │
│ Firing Rate │ Capacity │ Performance │ Loss                 │
│ [Tabs: Firing Rate | Capacity | Performance | Loss]         │
├─────────────────────────────────────────────────────────────┤
│ 🌱 Growth History                                            │
│ Neuron Count Over Time       │ Capacity by Region           │
│ Recent Growth Events Table                                   │
├─────────────────────────────────────────────────────────────┤
│ 💤 Consolidation Timeline                                    │
│ Consolidation Frequency      │ Statistics                   │
│ Recent Consolidation Events                                  │
├─────────────────────────────────────────────────────────────┤
│ ✅ Milestone Checklist                                       │
│ Total │ Passed │ Completion                                  │
│ [========= Milestone Progress Bar =========]                │
│ ✅ Milestone 1 ❌ Milestone 2 ✅ Milestone 3               │
├─────────────────────────────────────────────────────────────┤
│ ℹ️ Training Information                                     │
│ Duration │ Growth Events │ Consolidations                   │
└─────────────────────────────────────────────────────────────┘
```

## Data Sources

The dashboard reads training state from checkpoint files:

### Required Files

1. **Checkpoint Metadata** (`metadata_*.json`):
   ```json
   {
     "current_stage": 0,
     "step": 50000,
     "current_week": 2,
     "total_weeks": 4,
     "metrics": {
       "firing_rate": 0.112,
       "capacity": 0.75,
       "performance": 0.92,
       "loss": 0.234
     },
     "milestones": {
       "mnist_accuracy": true,
       "sequence_prediction": true,
       "phoneme_discrimination": false
     },
     "warnings": []
   }
   ```

2. **Training Log** (`training_log.json`):
   ```json
   {"step": 1000, "firing_rate": 0.11, "capacity": 0.65, ...}
   {"step": 2000, "event_type": "growth", "region": "cortex", ...}
   {"step": 3000, "event_type": "consolidation", "patterns_replayed": 150, ...}
   ```

### Data Format

The dashboard expects JSON-formatted logs with the following structure:

**Metrics Log Entry**:
- `step`: Training step number
- `firing_rate`: Average firing rate (0-1)
- `capacity`: System capacity (0-1)
- `performance`: Task accuracy (0-1)
- `loss`: Training loss (float)
- `neuron_counts`: Dict of region → neuron count
- `capacity_by_region`: Dict of region → capacity

**Growth Event Entry**:
- `event_type`: "growth"
- `step`: Training step
- `region`: Brain region name
- `neurons_added`: Number of neurons added
- `new_total`: New total neuron count
- `reason`: Reason for growth

**Consolidation Event Entry**:
- `event_type`: "consolidation"
- `step`: Training step
- `sleep_stage`: "NREM" or "REM"
- `patterns_replayed`: Number of patterns
- `duration_steps`: Duration in steps

## Integration with CurriculumTrainer

The dashboard is designed to work with `CurriculumTrainer` and `CurriculumLogger`:

```python
from thalia.training import CurriculumTrainer, CurriculumLogger

# Initialize trainer with logger
logger = CurriculumLogger(
    log_dir="checkpoints/curriculum",
    console_log=True,
    file_log=True,
    json_log=True  # Required for dashboard
)

trainer = CurriculumTrainer(
    brain=brain,
    checkpoint_dir="checkpoints/curriculum",
    logger=logger
)

# Train stages
trainer.train_stage(stage=0, config=stage_config)
```

The logger will automatically create the required JSON files for the dashboard.

## Monitoring Long Training Runs

For multi-hour or multi-day training runs:

1. **Start Training** in one terminal:
   ```bash
   python examples/curriculum_training_example.py
   ```

2. **Launch Dashboard** in another terminal:
   ```bash
   streamlit run examples/curriculum_dashboard.py
   ```

3. **Enable Auto-Refresh** in the sidebar with 5-10 second intervals

4. **Monitor Progress** in real-time without interrupting training

## Performance Considerations

- **File I/O**: Dashboard reads files on each refresh (keep refresh rate reasonable)
- **History Size**: Large training logs may slow rendering (dashboard shows recent data)
- **Checkpoint Size**: Checkpoint loading is fast (metadata only, not full model)
- **Browser Performance**: Modern browsers handle Streamlit efficiently

## Troubleshooting

### Dashboard Not Loading

```bash
# Check Streamlit is installed
streamlit --version

# Re-install if needed
pip install --upgrade streamlit
```

### No Data Displayed

- Verify checkpoint directory exists and contains `metadata_*.json` files
- Check that `CurriculumLogger` has `json_log=True` enabled
- Ensure training has progressed enough to generate checkpoints

### Slow Refresh

- Reduce auto-refresh rate (10-30 seconds)
- Disable detailed sections in sidebar
- Trim old log files if very large

### Port Already in Use

```bash
# Use custom port
streamlit run examples/curriculum_dashboard.py --server.port 8502
```

## Advanced Usage

### Custom Metrics

Extend the dashboard by modifying `load_training_state()` to load additional metrics from your logs.

### Multiple Training Runs

Monitor multiple training runs by launching multiple dashboard instances on different ports:

```bash
streamlit run examples/curriculum_dashboard.py --server.port 8501
streamlit run examples/curriculum_dashboard.py --server.port 8502
streamlit run examples/curriculum_dashboard.py --server.port 8503
```

Configure each with different checkpoint directories.

### Export Data

Use pandas to export dashboard data:

```python
# In dashboard code
state = load_training_state(checkpoint_dir)
state.metrics_history.to_csv('metrics_export.csv')
```

## Screenshots

(Add screenshots here after implementation)

## Next Steps

1. **Enhance Visualizations**: Add plotly for interactive charts
2. **Add Comparison Mode**: Compare multiple training runs side-by-side
3. **Real-Time Streaming**: WebSocket connection for true real-time updates
4. **Alert System**: Email/Slack notifications for warnings
5. **Configuration UI**: Edit training parameters from dashboard

## Contributing

Suggestions for dashboard improvements:
- Additional metric visualizations
- Better health check displays
- Mobile-responsive layout
- Dark mode theme
- Export functionality

## License

Same as Thalia project license.

---

**Last Updated**: December 8, 2025
