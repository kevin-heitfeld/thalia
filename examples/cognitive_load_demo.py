"""
Cognitive Load Monitoring Demo.

This script demonstrates the CognitiveLoadMonitor system that prevents
mechanism overload during curriculum stage transitions.

**Demonstrations**:
1. Basic load monitoring with single mechanisms
2. Multi-mechanism load tracking
3. Overload detection and deactivation suggestions
4. Priority-based deactivation
5. Load statistics and analysis

**Usage**:
    python examples/cognitive_load_demo.py

**Expected Behavior**:
- Monitor shows current load, headroom, and active mechanisms
- When load exceeds 90%, system suggests deactivations
- Priority-based suggestions (LOW/MEDIUM before HIGH/CRITICAL)
- Load statistics show min/max/mean over time

**Output**:
- Console output showing status reports
- Demonstrates all cognitive load monitoring features
"""

from __future__ import annotations

from thalia.training import (
    CognitiveLoadMonitor,
    MechanismPriority,
    CurriculumStage,
)


def demo_basic_load_monitoring():
    """Demo 1: Basic load monitoring with single mechanisms."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Load Monitoring")
    print("=" * 80)
    print("\nScenario: Adding mechanisms one-by-one and monitoring load\n")
    
    # Create monitor with default 90% threshold
    monitor = CognitiveLoadMonitor(load_threshold=0.9)
    
    # Add mechanisms incrementally
    print("Adding visual processing (0.2 load, CRITICAL)...")
    monitor.add_mechanism(
        'visual_processing',
        cost=0.2,
        priority=MechanismPriority.CRITICAL,
        stage_introduced=CurriculumStage.SENSORIMOTOR,
        can_deactivate=False,  # Core perceptual system
    )
    print(f"Current load: {monitor.calculate_load():.2f}")
    print(f"Headroom: {monitor.get_headroom():.2f}")
    print(f"Overloaded: {monitor.is_overloaded()}")
    
    print("\nAdding working memory (0.3 load, HIGH)...")
    monitor.add_mechanism(
        'working_memory',
        cost=0.3,
        priority=MechanismPriority.HIGH,
        stage_introduced=CurriculumStage.PHONOLOGY,
    )
    print(f"Current load: {monitor.calculate_load():.2f}")
    print(f"Headroom: {monitor.get_headroom():.2f}")
    print(f"Overloaded: {monitor.is_overloaded()}")
    
    print("\nAdding language processing (0.4 load, HIGH)...")
    monitor.add_mechanism(
        'language_processing',
        cost=0.4,
        priority=MechanismPriority.HIGH,
        stage_introduced=CurriculumStage.TODDLER,
    )
    print(f"Current load: {monitor.calculate_load():.2f}")
    print(f"Headroom: {monitor.get_headroom():.2f}")
    print(f"Overloaded: {monitor.is_overloaded()}")
    
    # Show status report
    print("\n" + monitor.get_status_report())


def demo_multi_mechanism_tracking():
    """Demo 2: Multi-mechanism load tracking with all priorities."""
    print("\n" + "=" * 80)
    print("DEMO 2: Multi-Mechanism Load Tracking")
    print("=" * 80)
    print("\nScenario: Full cognitive system with all priority levels\n")
    
    monitor = CognitiveLoadMonitor(load_threshold=0.9)
    
    # CRITICAL: Core perceptual systems (cannot deactivate)
    monitor.add_mechanism(
        'visual_processing',
        cost=0.15,
        priority=MechanismPriority.CRITICAL,
        stage_introduced=CurriculumStage.SENSORIMOTOR,
        can_deactivate=False,
    )
    monitor.add_mechanism(
        'auditory_processing',
        cost=0.10,
        priority=MechanismPriority.CRITICAL,
        stage_introduced=CurriculumStage.SENSORIMOTOR,
        can_deactivate=False,
    )
    
    # HIGH: Core learning mechanisms for current stage
    monitor.add_mechanism(
        'working_memory',
        cost=0.25,
        priority=MechanismPriority.HIGH,
        stage_introduced=CurriculumStage.PHONOLOGY,
    )
    monitor.add_mechanism(
        'language_processing',
        cost=0.20,
        priority=MechanismPriority.HIGH,
        stage_introduced=CurriculumStage.TODDLER,
    )
    
    # MEDIUM: Supporting mechanisms
    monitor.add_mechanism(
        'episodic_memory',
        cost=0.15,
        priority=MechanismPriority.MEDIUM,
        stage_introduced=CurriculumStage.TODDLER,
    )
    
    # LOW: Optional enhancements
    monitor.add_mechanism(
        'attention_modulation',
        cost=0.10,
        priority=MechanismPriority.LOW,
        stage_introduced=CurriculumStage.READING,
    )
    
    print(monitor.get_status_report())
    
    # Show load breakdown by priority
    print("\nLoad Breakdown by Priority:")
    breakdown = monitor.get_load_by_priority()
    for priority, load in breakdown.items():
        print(f"  {priority.name:8}: {load:.2f}")
    
    # Show load breakdown by stage
    print("\nLoad Breakdown by Stage:")
    stage_breakdown = monitor.get_load_by_stage()
    for stage, load in sorted(stage_breakdown.items(), key=lambda x: (x[0] is None, x[0])):
        stage_name = stage.name if stage else "None"
        print(f"  {stage_name:20}: {load:.2f}")


def demo_overload_detection():
    """Demo 3: Overload detection and deactivation suggestions."""
    print("\n" + "=" * 80)
    print("DEMO 3: Overload Detection & Deactivation Suggestions")
    print("=" * 80)
    print("\nScenario: System becomes overloaded during stage transition\n")
    
    monitor = CognitiveLoadMonitor(load_threshold=0.9)
    
    # Start with existing mechanisms
    print("Stage 2 (Toddler) mechanisms:")
    monitor.add_mechanism('visual_processing', cost=0.15, priority=MechanismPriority.CRITICAL, can_deactivate=False)
    monitor.add_mechanism('auditory_processing', cost=0.10, priority=MechanismPriority.CRITICAL, can_deactivate=False)
    monitor.add_mechanism('working_memory', cost=0.25, priority=MechanismPriority.HIGH)
    monitor.add_mechanism('language_processing', cost=0.20, priority=MechanismPriority.HIGH)
    print(f"  Load: {monitor.calculate_load():.2f} / {monitor.load_threshold:.2f}")
    print(f"  Status: {'OVERLOADED' if monitor.is_overloaded() else 'OK'}")
    
    # Transition to Stage 3 (Preschool) - adds new mechanisms
    print("\nTransitioning to Stage 3 (Preschool)...")
    print("Adding new stage mechanisms:")
    
    monitor.add_mechanism('episodic_memory', cost=0.15, priority=MechanismPriority.MEDIUM)
    print(f"  + episodic_memory (0.15) -> Load: {monitor.calculate_load():.2f}")
    
    monitor.add_mechanism('theory_of_mind', cost=0.12, priority=MechanismPriority.HIGH)
    print(f"  + theory_of_mind (0.12) -> Load: {monitor.calculate_load():.2f}")
    
    monitor.add_mechanism('narrative_construction', cost=0.10, priority=MechanismPriority.MEDIUM)
    print(f"  + narrative_construction (0.10) -> Load: {monitor.calculate_load():.2f}")
    
    # Check for overload
    print(f"\nFinal load: {monitor.calculate_load():.2f} / {monitor.load_threshold:.2f}")
    print(f"Overloaded: {monitor.is_overloaded()}")
    
    if monitor.is_overloaded():
        print("\n" + "!" * 60)
        print("COGNITIVE OVERLOAD DETECTED!")
        print("!" * 60)
        
        # Get single deactivation suggestion
        suggestion = monitor.suggest_deactivation()
        if suggestion:
            print(f"\nImmediate suggestion: Deactivate '{suggestion}'")
        
        # Get multiple deactivation suggestions
        suggestions = monitor.suggest_multiple_deactivations(target_load=0.8)
        print(f"\nTo reach 80% load, deactivate (in order):")
        for i, name in enumerate(suggestions, 1):
            print(f"  {i}. {name}")
    
    print("\n" + monitor.get_status_report())


def demo_priority_based_deactivation():
    """Demo 4: Priority-based deactivation during overload."""
    print("\n" + "=" * 80)
    print("DEMO 4: Priority-Based Deactivation")
    print("=" * 80)
    print("\nScenario: Systematically deactivate mechanisms by priority\n")
    
    monitor = CognitiveLoadMonitor(load_threshold=0.9)
    
    # Create overloaded system
    monitor.add_mechanism('visual', cost=0.15, priority=MechanismPriority.CRITICAL, can_deactivate=False)
    monitor.add_mechanism('auditory', cost=0.10, priority=MechanismPriority.CRITICAL, can_deactivate=False)
    monitor.add_mechanism('working_memory', cost=0.25, priority=MechanismPriority.HIGH)
    monitor.add_mechanism('language', cost=0.20, priority=MechanismPriority.HIGH)
    monitor.add_mechanism('episodic_memory', cost=0.15, priority=MechanismPriority.MEDIUM)
    monitor.add_mechanism('narrative', cost=0.10, priority=MechanismPriority.MEDIUM)
    monitor.add_mechanism('attention_mod', cost=0.08, priority=MechanismPriority.LOW)
    monitor.add_mechanism('curiosity_drive', cost=0.05, priority=MechanismPriority.LOW)
    
    print(f"Initial load: {monitor.calculate_load():.2f}")
    print(monitor.get_status_report())
    
    # Deactivate mechanisms until load is acceptable
    print("\nDeactivating mechanisms to reduce load...\n")
    iteration = 0
    while monitor.is_overloaded() and iteration < 10:
        iteration += 1
        suggestion = monitor.suggest_deactivation()
        if not suggestion:
            print("No more deactivatable mechanisms!")
            break
        
        # Find mechanism details before deactivating
        mech = next(m for m in monitor.active_mechanisms if m.name == suggestion)
        
        # Deactivate
        success = monitor.deactivate_mechanism(suggestion)
        if success:
            print(f"Iteration {iteration}:")
            print(f"  Deactivated: {suggestion} (Priority: {mech.priority.name}, Cost: {mech.cost:.2f})")
            print(f"  New load: {monitor.calculate_load():.2f}")
            print(f"  Status: {'OVERLOADED' if monitor.is_overloaded() else 'OK'}")
            print()
    
    print("\n" + monitor.get_status_report())
    
    # Reactivate LOW priority mechanisms
    print("\nReactivating LOW priority mechanisms...\n")
    monitor.reactivate_mechanism('curiosity_drive')
    monitor.reactivate_mechanism('attention_mod')
    
    print(f"After reactivation:")
    print(f"  Load: {monitor.calculate_load():.2f}")
    print(f"  Status: {'OVERLOADED' if monitor.is_overloaded() else 'OK'}")


def demo_load_statistics():
    """Demo 5: Load statistics and analysis over time."""
    print("\n" + "=" * 80)
    print("DEMO 5: Load Statistics & Analysis")
    print("=" * 80)
    print("\nScenario: Monitor load changes during training session\n")
    
    import time
    
    monitor = CognitiveLoadMonitor(load_threshold=0.9)
    
    # Simulate training session with changing load
    print("Simulating training session...\n")
    
    # Phase 1: Start with core mechanisms
    print("Phase 1: Core mechanisms (0-10s)")
    monitor.add_mechanism('visual', cost=0.15, priority=MechanismPriority.CRITICAL, can_deactivate=False)
    monitor.add_mechanism('auditory', cost=0.10, priority=MechanismPriority.CRITICAL, can_deactivate=False)
    monitor.add_mechanism('working_memory', cost=0.25, priority=MechanismPriority.HIGH)
    print(f"  Load: {monitor.calculate_load():.2f}")
    time.sleep(0.1)
    
    # Phase 2: Add language processing
    print("\nPhase 2: Add language (10-20s)")
    monitor.add_mechanism('language', cost=0.20, priority=MechanismPriority.HIGH)
    print(f"  Load: {monitor.calculate_load():.2f}")
    time.sleep(0.1)
    
    # Phase 3: Add memory systems
    print("\nPhase 3: Add memory systems (20-30s)")
    monitor.add_mechanism('episodic_memory', cost=0.15, priority=MechanismPriority.MEDIUM)
    print(f"  Load: {monitor.calculate_load():.2f}")
    time.sleep(0.1)
    
    # Phase 4: Add optional enhancements (causes overload)
    print("\nPhase 4: Add enhancements (30-40s)")
    monitor.add_mechanism('attention_mod', cost=0.12, priority=MechanismPriority.LOW)
    print(f"  Load: {monitor.calculate_load():.2f} - OVERLOADED!")
    time.sleep(0.1)
    
    # Phase 5: Deactivate to recover
    print("\nPhase 5: Deactivate to recover (40-50s)")
    monitor.deactivate_mechanism('attention_mod')
    print(f"  Load: {monitor.calculate_load():.2f}")
    time.sleep(0.1)
    
    # Get statistics
    stats = monitor.get_load_statistics()
    print("\n" + "=" * 60)
    print("Load Statistics (Full Session)")
    print("=" * 60)
    print(f"Minimum load: {stats['min']:.2f}")
    print(f"Maximum load: {stats['max']:.2f}")
    print(f"Mean load: {stats['mean']:.2f}")
    print(f"Current load: {stats['current']:.2f}")
    print("=" * 60)
    
    # Show final status
    print("\n" + monitor.get_status_report())


def main():
    """Run all cognitive load monitoring demonstrations."""
    print("\n" + "=" * 80)
    print("COGNITIVE LOAD MONITORING DEMONSTRATIONS")
    print("=" * 80)
    print("\nThis script demonstrates the CognitiveLoadMonitor system")
    print("for preventing mechanism overload during stage transitions.")
    
    # Run all demos
    demo_basic_load_monitoring()
    demo_multi_mechanism_tracking()
    demo_overload_detection()
    demo_priority_based_deactivation()
    demo_load_statistics()
    
    # Summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Monitor tracks cognitive load from active mechanisms")
    print("  2. Load threshold (default 90%) prevents overload")
    print("  3. Priority-based deactivation (LOW → MEDIUM → HIGH → CRITICAL)")
    print("  4. CRITICAL mechanisms cannot be deactivated")
    print("  5. Statistics track load changes over time")
    print("\nIntegration with CurriculumTrainer:")
    print("  - Monitor active during stage transitions")
    print("  - Suggests mechanism deactivations when overloaded")
    print("  - Adjusts old stage review ratios automatically")
    print("  - Prevents cognitive overload and catastrophic forgetting")
    print()


if __name__ == "__main__":
    main()
