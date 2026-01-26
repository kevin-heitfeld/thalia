# Configuration Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2026-01-26 14:17:33
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all configuration dataclasses in Thalia.

Total: 0 configuration classes

## Configuration Classes

## 💡 Configuration Best Practices

### Common Patterns

1. **Start with defaults**: All configs have biologically-motivated defaults
2. **Override selectively**: Only change what's needed for your task
3. **Validate early**: Use config validation before training
4. **Document changes**: Keep notes on why you changed defaults

### Performance Considerations

- **Layer sizes**: Larger = more capacity but slower training
- **Sparsity**: Higher sparsity = faster but less connectivity
- **Learning rates**: Too high = instability, too low = slow learning
- **Time constants**: Should match biological ranges (ms scale)

### Common Pitfalls

⚠️ **Too small networks**: Use at least 64 neurons per region

⚠️ **Mismatched time scales**: Keep tau values in biological range (5-200ms)

⚠️ **Extreme learning rates**: Stay within 0.0001-0.01 range

⚠️ **Disabled plasticity**: Ensure learning strategies are enabled

## 📚 Related Documentation

- [COMPONENT_CATALOG.md](COMPONENT_CATALOG.md) - Components using these configs
- [LEARNING_STRATEGIES_API.md](LEARNING_STRATEGIES_API.md) - Learning rules and parameters
- [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - Configuration examples
- [ENUMERATIONS_REFERENCE.md](ENUMERATIONS_REFERENCE.md) - Enum types used in configs

