# Implementation Patterns

Common patterns and best practices for working with the Thalia codebase.

## Available Patterns

### ⭐ [Component Parity](./component-parity.md) - READ THIS FIRST
Regions and pathways are equals - both implement the BrainComponent protocol.
- Pathways are NOT just "glue" - they are active learning components
- When adding features to regions, MUST add to pathways too
- Unified interface prevents forgetting pathways

### [Configuration](./configuration.md)
Config hierarchy and parameter management

### [State Management](./state-management.md)
When to use RegionState vs attributes

### [Mixins](./mixins.md)
Available mixins and their methods

## Usage

These documents provide implementation guidance for:
- Creating new brain regions **and pathways**
- Managing neuron state and learning rules
- Applying diagnostic and action selection capabilities
- Configuring training and evaluation

## Related Documentation

- **[Design Docs](../design/)** - Detailed design specifications
- **[Architecture](../architecture/)** - High-level system architecture
