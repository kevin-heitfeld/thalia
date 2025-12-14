# ADR-007: PyTorch Consistency - Use forward() Instead of encode()/decode()

**Date**: December 8, 2025
**Status**: Accepted
**Context**: ADR-005 cleanup revealed non-standard method names

## Context

During ADR-005 migration (removing batch dimensions), we discovered that sensory pathways used `encode()` instead of the standard PyTorch `forward()` method. Similarly, the `LanguageDecoder` had both `decode()` and `forward()`, causing redundancy.

### Problem

```python
# ❌ Non-standard (before):
class RetinalEncoder(SensoryPathway):
    def encode(self, image):  # Custom name
        ...

class LanguageDecoder(nn.Module):
    def decode(self, spikes):  # Redundant
        ...
    def forward(self, spikes):  # Just calls decode()
        return self.decode(spikes)
```

**Issues:**
1. **Breaks PyTorch conventions**: `nn.Module` subclasses should use `forward()`
2. **Inconsistent with regions**: Neural components use `forward()`
3. **Redundant code**: Decoder had both methods
4. **Confusing for users**: Unclear which method to call

## Decision

**Rename all `encode()` → `forward()` and consolidate `decode()` → `forward()`**

### Rationale

1. **PyTorch standard**: `forward()` enables `model(input)` callable syntax
2. **Consistency**: All components (regions, pathways, sensory encoders) inherit from NeuralComponent and use `forward()`
3. **Simplicity**: One method name across entire codebase
4. **Type checking**: PyTorch's type system expects `forward()`

## Implementation

### Changes Made

#### 1. **SensoryPathway Base Class** (`src/thalia/sensory/pathways.py`)

```python
# BEFORE:
class SensoryPathway(nn.Module):
    @abstractmethod
    def encode(self, raw_input, **kwargs):
        """Convert raw input to spike patterns."""
        pass

# AFTER:
class SensoryPathway(nn.Module):
    @abstractmethod
    def forward(self, raw_input, **kwargs):
        """Convert raw input to spike patterns (standard PyTorch)."""
        pass
```

#### 2. **RetinalEncoder** (`src/thalia/sensory/pathways.py`)

```python
# BEFORE:
class RetinalEncoder(SensoryPathway):
    def encode(self, image, reset_adaptation=False):
        ...

# AFTER:
class RetinalEncoder(SensoryPathway):
    def forward(self, image, reset_adaptation=False):
        ...
```

#### 3. **AuditoryPathway** (`src/thalia/sensory/pathways.py`)

```python
# BEFORE:
class AuditoryPathway(SensoryPathway):
    def encode(self, audio, **kwargs):
        ...

# AFTER:
class AuditoryPathway(SensoryPathway):
    def forward(self, audio, **kwargs):
        ...
```

#### 4. **LanguagePathway** (`src/thalia/sensory/pathways.py`)

```python
# BEFORE:
class LanguagePathway(SensoryPathway):
    def encode(self, raw_input, position_ids=None, **kwargs):
        ...

# AFTER:
class LanguagePathway(SensoryPathway):
    def forward(self, raw_input, position_ids=None, **kwargs):
        ...
```

#### 5. **LanguageDecoder** (`src/thalia/language/decoder.py`)

```python
# BEFORE (redundant):
class LanguageDecoder(nn.Module):
    def decode(self, spikes):
        features = self._integrate_spikes(spikes)
        logits = self._decode_features(features)
        return logits / self.config.temperature

    def forward(self, spikes, return_features=False):
        logits = self.decode(spikes)  # Just calls decode!
        if return_features:
            features = self._integrate_spikes(spikes)
            return logits, features
        return logits

# AFTER (consolidated):
class LanguageDecoder(nn.Module):
    def forward(self, spikes, return_features=False):
        """Decode spikes to token logits/probabilities."""
        features = self._integrate_spikes(spikes)
        logits = self._decode_features(features)
        logits = logits / self.config.temperature

        if return_features:
            return logits, features
        return logits
```

### Files Modified

- `src/thalia/sensory/pathways.py`:
  - `SensoryPathway.encode()` → `SensoryPathway.forward()`
  - `RetinalEncoder.encode()` → `RetinalEncoder.forward()`
  - `CochlearEncoder.forward()` (already correct)
  - `AuditoryPathway.encode()` → `AuditoryPathway.forward()`
  - `LanguagePathway.encode()` → `LanguagePathway.forward()`

- `src/thalia/language/decoder.py`:
  - Removed `LanguageDecoder.decode()`
  - Consolidated into `LanguageDecoder.forward()`

## Benefits

### 1. **Standard PyTorch Usage**

```python
# ✅ Now works:
encoder = RetinalEncoder(config)
spikes, metadata = encoder(image)  # Callable syntax!

# Instead of:
# spikes, metadata = encoder.encode(image)  # Non-standard
```

### 2. **Consistent Across Codebase**

```python
# All components use forward():
cortex_spikes = cortex(input_spikes)  # NeuralComponent (region)
pathway_spikes = pathway(cortex_spikes)  # NeuralComponent (pathway)
sensory_spikes = encoder(image)  # SensoryPathway
logits = decoder(sensory_spikes)  # LanguageDecoder
```

### 3. **Type Checker Happy**

```python
# PyTorch's type system expects forward():
model: nn.Module = encoder
output = model(input)  # ✅ Type checker knows this works
```

### 4. **Less Code**

- Removed redundant `LanguageDecoder.decode()` method
- ~20 lines of duplicate code eliminated

## Migration Guide

### For External Users

If you have code using the old API:

```python
# OLD:
spikes, metadata = retinal_encoder.encode(image)
logits = decoder.decode(spikes)

# NEW:
spikes, metadata = retinal_encoder(image)
logits = decoder(spikes)
```

### For Internal Code

All internal uses already updated. No action needed.

## Testing

Created `temp/test_sensory_pathways_adr005.py` to verify:

- ✅ **RetinalEncoder**: `encoder(image)` works
- ✅ **AuditoryPathway**: `pathway(audio)` works
- ✅ **LanguagePathway**: `pathway(token)` works
- ✅ **All enforce ADR-005**: 1D inputs, 2D temporal outputs

## Related ADRs

- **ADR-005**: No batch dimension (this cleanup revealed naming inconsistency)
- **ADR-004**: Bool tensors for spikes

## Conclusion

**All components now use standard PyTorch `forward()` convention.**

This simplifies the API, improves consistency, and makes the codebase more maintainable.

---

**Approved by**: GitHub Copilot
**Implemented**: December 8, 2025
