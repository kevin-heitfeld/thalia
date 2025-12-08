"""
Streamlit Dashboard for Thalia Curriculum Training Monitoring

Real-time monitoring dashboard for curriculum training with:
- Stage progress tracking
- Live metrics display
- Growth history visualization
- Consolidation timeline
- Milestone checklist
- Health warnings

Usage:
    streamlit run examples/curriculum_dashboard.py -- --checkpoint-dir checkpoints/curriculum
"""

import streamlit as st
import torch
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# Streamlit page config
st.set_page_config(
    page_title="Thalia Curriculum Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .health-good {
        color: #28a745;
        font-weight: bold;
    }
    .health-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .health-critical {
        color: #dc3545;
        font-weight: bold;
    }
    .milestone-passed {
        color: #28a745;
    }
    .milestone-failed {
        color: #dc3545;
    }
    .stage-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)


class TrainingState:
    """Container for current training state loaded from checkpoints."""
    
    def __init__(self):
        self.current_stage: int = -1
        self.stage_name: str = "Unknown"
        self.current_week: int = 0
        self.total_weeks: int = 0
        self.stage_progress: float = 0.0
        self.current_step: int = 0
        self.total_steps: int = 0
        
        # Metrics
        self.firing_rate: float = 0.0
        self.firing_rate_delta: float = 0.0
        self.capacity: float = 0.0
        self.capacity_delta: float = 0.0
        self.performance: float = 0.0
        self.performance_delta: float = 0.0
        self.loss: float = 0.0
        self.loss_delta: float = 0.0
        
        # History data
        self.neuron_count_history: pd.DataFrame = pd.DataFrame()
        self.metrics_history: pd.DataFrame = pd.DataFrame()
        self.growth_events: List[Dict[str, Any]] = []
        self.consolidation_events: List[Dict[str, Any]] = []
        self.capacity_by_region: Dict[str, float] = {}
        
        # Milestones
        self.milestones: Dict[str, bool] = {}
        
        # Warnings
        self.warnings: List[str] = []
        
        # Metadata
        self.last_update: Optional[datetime] = None
        self.training_duration: timedelta = timedelta()


def load_training_state(checkpoint_dir: str) -> TrainingState:
    """
    Load training state from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        TrainingState object with current state
    """
    state = TrainingState()
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        st.warning(f"⚠️ Checkpoint directory not found: {checkpoint_dir}")
        return state
    
    # Load latest checkpoint metadata
    metadata_files = sorted(checkpoint_path.glob("metadata_*.json"))
    if not metadata_files:
        st.warning("⚠️ No checkpoint metadata found")
        return state
    
    latest_metadata = metadata_files[-1]
    
    try:
        with open(latest_metadata, 'r') as f:
            metadata = json.load(f)
        
        # Extract stage information
        state.current_stage = metadata.get('current_stage', -1)
        state.stage_name = get_stage_name(state.current_stage)
        state.current_step = metadata.get('step', 0)
        state.current_week = metadata.get('current_week', 0)
        state.total_weeks = metadata.get('total_weeks', 0)
        state.stage_progress = state.current_week / max(state.total_weeks, 1)
        
        # Extract metrics
        metrics = metadata.get('metrics', {})
        state.firing_rate = metrics.get('firing_rate', 0.0)
        state.capacity = metrics.get('capacity', 0.0)
        state.performance = metrics.get('performance', 0.0)
        state.loss = metrics.get('loss', 0.0)
        
        # Calculate deltas from previous checkpoint
        if len(metadata_files) > 1:
            with open(metadata_files[-2], 'r') as f:
                prev_metadata = json.load(f)
            prev_metrics = prev_metadata.get('metrics', {})
            state.firing_rate_delta = state.firing_rate - prev_metrics.get('firing_rate', 0.0)
            state.capacity_delta = state.capacity - prev_metrics.get('capacity', 0.0)
            state.performance_delta = state.performance - prev_metrics.get('performance', 0.0)
            state.loss_delta = state.loss - prev_metrics.get('loss', 0.0)
        
        # Load history data
        state = load_history_data(checkpoint_path, state)
        
        # Load milestones
        state.milestones = metadata.get('milestones', {})
        
        # Load warnings
        state.warnings = metadata.get('warnings', [])
        
        # Metadata
        state.last_update = datetime.fromtimestamp(latest_metadata.stat().st_mtime)
        state.training_duration = timedelta(seconds=metadata.get('training_time_seconds', 0))
        
    except Exception as e:
        st.error(f"❌ Error loading checkpoint: {e}")
    
    return state


def load_history_data(checkpoint_path: Path, state: TrainingState) -> TrainingState:
    """Load historical data from checkpoint logs."""
    
    # Load training log if available
    log_file = checkpoint_path / "training_log.json"
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                logs = [json.loads(line) for line in f]
            
            # Build metrics history
            steps = [log['step'] for log in logs]
            state.metrics_history = pd.DataFrame({
                'step': steps,
                'firing_rate': [log.get('firing_rate', 0) for log in logs],
                'capacity': [log.get('capacity', 0) for log in logs],
                'performance': [log.get('performance', 0) for log in logs],
                'loss': [log.get('loss', 0) for log in logs],
            })
            
            # Extract growth events
            state.growth_events = [
                log for log in logs if log.get('event_type') == 'growth'
            ]
            
            # Extract consolidation events
            state.consolidation_events = [
                log for log in logs if log.get('event_type') == 'consolidation'
            ]
            
            # Build neuron count history
            neuron_counts = {}
            for log in logs:
                if 'neuron_counts' in log:
                    for region, count in log['neuron_counts'].items():
                        if region not in neuron_counts:
                            neuron_counts[region] = []
                        neuron_counts[region].append(count)
            
            if neuron_counts:
                neuron_data = {'step': steps}
                neuron_data.update(neuron_counts)
                state.neuron_count_history = pd.DataFrame(neuron_data)
            
            # Get latest capacity by region
            if logs:
                state.capacity_by_region = logs[-1].get('capacity_by_region', {})
                
        except Exception as e:
            st.warning(f"⚠️ Could not load training log: {e}")
    
    return state


def get_stage_name(stage: int) -> str:
    """Map stage number to name."""
    stage_names = {
        -1: "Sensorimotor",
        0: "Sensory Foundations",
        1: "Object Permanence & Working Memory",
        2: "Language & Executive Function",
        3: "Reading & Planning",
        4: "Abstract Reasoning",
        5: "Expert Knowledge",
        6: "LLM-Level Capabilities"
    }
    return stage_names.get(stage, f"Stage {stage}")


def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Checkpoint directory
        checkpoint_dir = st.text_input(
            "Checkpoint Directory",
            value="checkpoints/curriculum",
            help="Path to the checkpoint directory"
        )
        
        # Refresh settings
        st.subheader("Refresh Settings")
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        refresh_rate = st.slider(
            "Refresh Rate (seconds)",
            min_value=1,
            max_value=30,
            value=5,
            disabled=not auto_refresh
        )
        
        # Display options
        st.subheader("Display Options")
        show_detailed_metrics = st.checkbox("Show Detailed Metrics", value=True)
        show_growth_details = st.checkbox("Show Growth Details", value=True)
        show_consolidation_details = st.checkbox("Show Consolidation Details", value=True)
        
        # Manual refresh button
        if st.button("🔄 Refresh Now", type="primary"):
            st.rerun()
        
        return {
            'checkpoint_dir': checkpoint_dir,
            'auto_refresh': auto_refresh,
            'refresh_rate': refresh_rate,
            'show_detailed_metrics': show_detailed_metrics,
            'show_growth_details': show_growth_details,
            'show_consolidation_details': show_consolidation_details,
        }


def render_header(state: TrainingState):
    """Render main header with stage information."""
    st.title("🧠 Thalia Curriculum Training Monitor")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Stage",
            f"Stage {state.current_stage}",
            state.stage_name
        )
    
    with col2:
        st.metric(
            "Week Progress",
            f"{state.current_week}/{state.total_weeks}",
            f"{state.stage_progress:.1%}"
        )
    
    with col3:
        st.metric(
            "Training Step",
            f"{state.current_step:,}",
            ""
        )
    
    with col4:
        if state.last_update:
            time_ago = datetime.now() - state.last_update
            st.metric(
                "Last Update",
                f"{time_ago.seconds}s ago",
                state.last_update.strftime("%H:%M:%S")
            )
    
    # Stage progress bar
    st.progress(state.stage_progress, text=f"Stage {state.current_stage} Progress: {state.stage_progress:.1%}")


def render_metrics_section(state: TrainingState, show_detailed: bool):
    """Render real-time metrics section."""
    st.header("📊 Current Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Firing Rate",
            f"{state.firing_rate:.3f}",
            f"{state.firing_rate_delta:+.3f}",
            delta_color="off"
        )
    
    with col2:
        st.metric(
            "Capacity",
            f"{state.capacity:.1%}",
            f"{state.capacity_delta:+.1%}",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Performance",
            f"{state.performance:.2%}",
            f"{state.performance_delta:+.2%}",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            "Loss",
            f"{state.loss:.4f}",
            f"{state.loss_delta:+.4f}",
            delta_color="inverse"
        )
    
    # Detailed metrics over time
    if show_detailed and not state.metrics_history.empty:
        st.subheader("Metrics History")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Firing Rate", "Capacity", "Performance", "Loss"])
        
        with tab1:
            st.line_chart(state.metrics_history[['step', 'firing_rate']].set_index('step'))
        
        with tab2:
            st.line_chart(state.metrics_history[['step', 'capacity']].set_index('step'))
        
        with tab3:
            st.line_chart(state.metrics_history[['step', 'performance']].set_index('step'))
        
        with tab4:
            st.line_chart(state.metrics_history[['step', 'loss']].set_index('step'))


def render_growth_section(state: TrainingState, show_details: bool):
    """Render growth history section."""
    st.header("🌱 Growth History")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Neuron Count Over Time")
        if not state.neuron_count_history.empty:
            st.line_chart(state.neuron_count_history.set_index('step'))
        else:
            st.info("No neuron count data available yet")
    
    with col2:
        st.subheader("Capacity by Region")
        if state.capacity_by_region:
            capacity_df = pd.DataFrame([
                {'Region': region, 'Capacity': capacity}
                for region, capacity in state.capacity_by_region.items()
            ])
            st.bar_chart(capacity_df.set_index('Region'))
        else:
            st.info("No capacity data available yet")
    
    # Growth events details
    if show_details and state.growth_events:
        st.subheader("Recent Growth Events")
        
        # Show last 10 growth events
        recent_events = state.growth_events[-10:]
        events_data = []
        for event in recent_events:
            events_data.append({
                'Step': event.get('step', 0),
                'Region': event.get('region', 'Unknown'),
                'Neurons Added': event.get('neurons_added', 0),
                'Reason': event.get('reason', 'N/A'),
                'New Total': event.get('new_total', 0)
            })
        
        st.dataframe(pd.DataFrame(events_data), use_container_width=True)


def render_consolidation_section(state: TrainingState, show_details: bool):
    """Render consolidation timeline section."""
    st.header("💤 Consolidation Timeline")
    
    if not state.consolidation_events:
        st.info("No consolidation events recorded yet")
        return
    
    # Timeline visualization
    timeline_data = []
    for event in state.consolidation_events:
        timeline_data.append({
            'Step': event.get('step', 0),
            'Stage': event.get('sleep_stage', 'Unknown'),
            'Patterns': event.get('patterns_replayed', 0),
            'Duration': event.get('duration_steps', 0)
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Consolidation Frequency")
        st.line_chart(timeline_df[['Step', 'Patterns']].set_index('Step'))
    
    with col2:
        st.subheader("Consolidation Statistics")
        st.metric("Total Events", len(state.consolidation_events))
        st.metric("Total Patterns Replayed", timeline_df['Patterns'].sum())
        st.metric("Average Patterns/Event", f"{timeline_df['Patterns'].mean():.1f}")
    
    # Detailed event log
    if show_details:
        st.subheader("Recent Consolidation Events")
        recent_events = timeline_df.tail(10)
        st.dataframe(recent_events, use_container_width=True)


def render_milestones_section(state: TrainingState):
    """Render milestone checklist section."""
    st.header("✅ Milestone Checklist")
    
    if not state.milestones:
        st.info("No milestones defined for current stage")
        return
    
    # Calculate statistics
    total = len(state.milestones)
    passed = sum(1 for v in state.milestones.values() if v)
    completion = passed / total if total > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Milestones", total)
    
    with col2:
        st.metric("Passed", passed)
    
    with col3:
        st.metric("Completion", f"{completion:.1%}")
    
    # Progress bar
    st.progress(completion, text=f"Milestone Completion: {completion:.1%}")
    
    # Checklist
    st.subheader("Detailed Milestones")
    
    for milestone, passed in state.milestones.items():
        icon = "✅" if passed else "❌"
        color_class = "milestone-passed" if passed else "milestone-failed"
        st.markdown(
            f'<span class="{color_class}">{icon} {milestone}</span>',
            unsafe_allow_html=True
        )


def render_health_warnings(state: TrainingState):
    """Render health warnings section."""
    if not state.warnings:
        st.success("✅ All systems healthy - no warnings")
        return
    
    st.warning(f"⚠️ {len(state.warnings)} Health Warning(s)")
    
    for i, warning in enumerate(state.warnings, 1):
        # Classify warning severity
        if any(keyword in warning.lower() for keyword in ['critical', 'fatal', 'error']):
            st.error(f"{i}. {warning}")
        elif any(keyword in warning.lower() for keyword in ['warning', 'caution', 'high']):
            st.warning(f"{i}. {warning}")
        else:
            st.info(f"{i}. {warning}")


def render_training_info(state: TrainingState):
    """Render training information section."""
    st.header("ℹ️ Training Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Duration")
        hours = state.training_duration.total_seconds() / 3600
        st.write(f"{hours:.2f} hours")
    
    with col2:
        st.subheader("Growth Events")
        st.write(f"{len(state.growth_events)} events")
    
    with col3:
        st.subheader("Consolidations")
        st.write(f"{len(state.consolidation_events)} events")


def main():
    """Main dashboard application."""
    
    # Render sidebar and get config
    config = render_sidebar()
    
    # Load training state
    state = load_training_state(config['checkpoint_dir'])
    
    # Main content
    render_header(state)
    
    st.divider()
    
    # Health warnings (prominent if present)
    if state.warnings:
        render_health_warnings(state)
        st.divider()
    
    # Metrics section
    render_metrics_section(state, config['show_detailed_metrics'])
    
    st.divider()
    
    # Growth section
    render_growth_section(state, config['show_growth_details'])
    
    st.divider()
    
    # Consolidation section
    render_consolidation_section(state, config['show_consolidation_details'])
    
    st.divider()
    
    # Milestones section
    render_milestones_section(state)
    
    st.divider()
    
    # Training info
    render_training_info(state)
    
    # Footer
    st.divider()
    st.caption(f"Dashboard last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto-refresh logic
    if config['auto_refresh']:
        time.sleep(config['refresh_rate'])
        st.rerun()


if __name__ == "__main__":
    main()
