# Component Catalog

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-21 19:50:31
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all registered brain regions and pathways in the Thalia component registry.

## Contents

- [Registered Regions](#registered-regions)
- [Registered Pathways](#registered-pathways)

## Registered Regions

Total: 8 regions

### `cerebellum`

**Class**: `Cerebellum`

**Config Class**: `CerebellumConfig`

**Source**: `thalia\regions\cerebellum_region.py`

**Description**: Cerebellar region with supervised error-corrective learning.

---

### `cortex`

**Class**: `LayeredCortex`

**Aliases**: `layered_cortex`

**Config Class**: `LayeredCortexConfig`

**Source**: `thalia\regions\cortex\layered_cortex.py`

**Description**: Multi-layer cortical microcircuit with proper layer separation and routing.

---

### `hippocampus`

**Class**: `TrisynapticHippocampus`

**Aliases**: `trisynaptic`

**Config Class**: `HippocampusConfig`

**Source**: `thalia\regions\hippocampus\trisynaptic.py`

**Description**: Biologically-accurate hippocampus with DG→CA3→CA1 trisynaptic circuit.

---

### `multimodal_integration`

**Class**: `MultimodalIntegration`

**Config Class**: `None`

**Source**: `thalia\regions\multisensory.py`

**Description**: Multimodal integration region for cross-modal fusion.

---

### `predictive_cortex`

**Class**: `PredictiveCortex`

**Config Class**: `PredictiveCortexConfig`

**Source**: `thalia\regions\cortex\predictive_cortex.py`

**Description**: Layered cortex with integrated predictive coding.

---

### `prefrontal`

**Class**: `Prefrontal`

**Aliases**: `pfc`

**Config Class**: `PrefrontalConfig`

**Source**: `thalia\regions\prefrontal.py`

**Description**: Prefrontal cortex with dopamine-gated working memory.

---

### `striatum`

**Class**: `Striatum`

**Config Class**: `StriatumConfig`

**Source**: `thalia\regions\striatum\striatum.py`

**Description**: Striatal region with three-factor reinforcement learning.

---

### `thalamus`

**Class**: `ThalamicRelay`

**Aliases**: `thalamic_relay`

**Config Class**: `ThalamicRelayConfig`

**Source**: `thalia\regions\thalamus.py`

**Description**: Thalamic relay nucleus with burst/tonic modes and attentional gating.

---

## Registered Pathways

Total: 4 pathways

### `auditory`

**Class**: `AuditoryPathway`

**Config Class**: `None`

**Source**: `thalia\pathways\sensory_pathways.py`

**Description**: Complete auditory pathway from audio to cortical input.

---

### `axonal`

**Class**: `AxonalProjection`

**Aliases**: `axonal_projection, pure_axon`

**Config Class**: `None`

**Source**: `thalia\pathways\axonal_projection.py`

**Description**: Pure axonal transmission between brain regions.

---

### `language`

**Class**: `LanguagePathway`

**Config Class**: `None`

**Source**: `thalia\pathways\sensory_pathways.py`

**Description**: Language pathway for text/token input using temporal/latency coding.

---

### `visual`

**Class**: `VisualPathway`

**Config Class**: `None`

**Source**: `thalia\pathways\sensory_pathways.py`

**Description**: Complete visual pathway from image to cortical input.

---

