"""
Working Memory Demo

Demonstrates working memory with reverberating circuits:
- Loading and maintaining patterns
- Capacity limits
- Decay and refresh
"""

import torch
from thalia.memory import WorkingMemory, WorkingMemoryConfig, WorkingMemorySNN


def demo_basic_memory():
    """Demonstrate basic working memory operations."""
    print("=" * 60)
    print("Basic Working Memory Demo")
    print("=" * 60)
    
    # Create working memory with 5 slots
    config = WorkingMemoryConfig(
        n_slots=5,
        slot_size=50,
        reverb_strength=0.8,
        decay_rate=0.01,
    )
    memory = WorkingMemory(config)
    memory.reset_state(batch_size=1)
    
    print(f"Created working memory with {config.n_slots} slots")
    print(f"Each slot has {config.slot_size} neurons")
    
    # Load some patterns
    print("\n--- Loading Items ---")
    items = ["phone", "keys", "wallet"]
    
    for i, item in enumerate(items):
        pattern = torch.randn(1, 50)
        memory.load(i, pattern, label=item)
        print(f"Loaded '{item}' into slot {i}")
    
    # Process loads
    memory()
    
    # Check status
    print("\n--- Memory Status ---")
    status = memory.get_status()
    for slot in status["slots"]:
        if slot["label"]:
            print(f"Slot {slot['index']}: '{slot['label']}' (activity: {slot['activity_level']:.3f})")
        else:
            print(f"Slot {slot['index']}: empty")
    
    print(f"\nCapacity used: {status['capacity_used']:.0%}")
    
    # Clear one slot
    print("\n--- Clearing Slot 1 ---")
    memory.clear(1)
    memory()  # Process clear
    
    status = memory.get_status()
    for slot in status["slots"]:
        label = slot["label"] or "empty"
        print(f"Slot {slot['index']}: {label}")


def demo_memory_maintenance():
    """Demonstrate memory maintenance through reverberating activity."""
    print("\n" + "=" * 60)
    print("Memory Maintenance Demo")
    print("=" * 60)
    
    config = WorkingMemoryConfig(
        n_slots=3,
        slot_size=50,
        reverb_strength=0.9,  # Strong reverberation
        decay_rate=0.005,     # Slow decay
        noise_std=0.01,
    )
    memory = WorkingMemory(config)
    memory.reset_state(batch_size=1)
    
    # Load a strong pattern
    print("Loading pattern into slot 0...")
    pattern = torch.ones(1, 50) * 2.0  # Strong activation
    memory.load(0, pattern, label="important")
    
    # Run for several steps and track activity
    print("\n--- Running Memory ---")
    print("Step | Slot 0 Activity")
    print("-" * 25)
    
    for t in range(20):
        memory()
        activity = memory.slots[0]._activity_level
        
        if t % 4 == 0:
            bar = "█" * int(activity * 50)
            print(f"{t:4d} | {activity:.3f} {bar}")
    
    print("\nMemory maintains activity through recurrent connections.")


def demo_capacity_limits():
    """Demonstrate working memory capacity limits."""
    print("\n" + "=" * 60)
    print("Capacity Limits Demo")
    print("=" * 60)
    
    config = WorkingMemoryConfig(n_slots=7)  # 7 slots like human WM
    memory = WorkingMemory(config)
    memory.reset_state(batch_size=1)
    
    print(f"Working memory capacity: {config.n_slots} items")
    print("(Similar to human 7±2 capacity)")
    
    # Try to load items
    items = ["apple", "banana", "cat", "dog", "elephant", "fish", "grape", "house"]
    
    print("\n--- Loading Items ---")
    for i, item in enumerate(items):
        empty_slot = memory.find_empty_slot()
        
        if empty_slot is not None:
            pattern = torch.randn(1, config.slot_size)
            memory.load(empty_slot, pattern, label=item)
            memory()  # Process
            print(f"Loaded '{item}' into slot {empty_slot} ✓")
        else:
            print(f"Cannot load '{item}' - memory full! ✗")
    
    print(f"\nFinal status: {memory.get_status()['active_count']}/{config.n_slots} slots used")


def demo_refresh_mechanism():
    """Demonstrate refreshing to prevent decay."""
    print("\n" + "=" * 60)
    print("Attention Refresh Demo")
    print("=" * 60)
    
    config = WorkingMemoryConfig(
        n_slots=3,
        slot_size=30,
        reverb_strength=0.7,
        decay_rate=0.02,  # Moderate decay
    )
    memory = WorkingMemory(config)
    memory.reset_state(batch_size=1)
    
    # Load pattern
    pattern = torch.ones(1, 30) * 1.5
    memory.load(0, pattern, label="memory_item")
    memory()
    
    print("Loaded item. Running with periodic refresh...\n")
    
    refresh_interval = 10
    
    for t in range(40):
        # Refresh every 10 steps (simulating attention)
        if t > 0 and t % refresh_interval == 0:
            memory.refresh(0, strength=0.5)
            print(f"Step {t:2d}: REFRESHED - activity = {memory.slots[0]._activity_level:.3f}")
        else:
            memory()
            if t % 5 == 0:
                print(f"Step {t:2d}: activity = {memory.slots[0]._activity_level:.3f}")
    
    print("\nPeriodic attention refreshes help maintain memory content.")


def demo_full_network():
    """Demonstrate full working memory SNN with encoding/decoding."""
    print("\n" + "=" * 60)
    print("Working Memory SNN Demo")
    print("=" * 60)
    
    # Create network
    config = WorkingMemoryConfig(
        n_slots=4,
        slot_size=32,
        reverb_strength=0.8,
    )
    network = WorkingMemorySNN(
        input_size=16,
        output_size=8,
        config=config,
    )
    network.reset_state(batch_size=1)
    
    print(f"Network: {network.input_size} input → {config.n_slots} slots → {network.output_size} output")
    
    # Load an input
    print("\n--- Loading Input Pattern ---")
    x = torch.randn(1, 16)
    output, mem_activity = network(x, load_slot=0)
    print(f"Input encoded and loaded into slot 0")
    print(f"Memory activity shape: {mem_activity.shape}")
    print(f"Output spikes: {output.sum().item():.0f}")
    
    # Run without new input
    print("\n--- Running from Memory ---")
    for t in range(10):
        output, mem_activity = network()
        print(f"Step {t}: memory mean = {mem_activity.mean():.4f}, output spikes = {output.sum().item():.0f}")
    
    print("\nNetwork generates output from maintained memory.")


def main():
    """Run all demos."""
    demo_basic_memory()
    demo_memory_maintenance()
    demo_capacity_limits()
    demo_refresh_mechanism()
    demo_full_network()
    
    print("\n" + "=" * 60)
    print("All working memory demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
