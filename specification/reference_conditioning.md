# NRR Reference-Conditioned Rendering

> Phase 5: The differentiating feature - conditioning neural output on developer-supplied identity assets

---

## 1. Overview

Reference-conditioned rendering is NRR's key innovation. Instead of asking a model to generate "a face," the system asks it to generate **this specific character's face**, conditioned on developer-supplied reference assets.

This creates a distinction between:
- **Training data** - teaches model general capabilities
- **Runtime reference data** - conditions output for specific identities

---

## 2. Reference Types

| Category | Purpose |
|----------|---------|
| `facial_reference` | Character face conditioning |
| `material_reference` | Material appearance |
| `hair_reference` | Hair appearance |
| `skin_reference` | Skin appearance |
| `clothing_reference` | Clothing conditioning |
| `expression_reference` | Expression conditioning |
| `identity_embedding` | Learned identity vector |

Not all references are required. Model declares which it needs.

---

## 3. Reference File (.nrrref)

```
.nrrref (ZIP)
├── metadata.json
├── textures/
│   ├── face.dds, hair.dds, skin.dds, ...
├── embeddings/
│   └── identity_embedding.npz
└── provenance.json
```

---

## 4. Identity Embedding

Learned vector capturing visual identity essence (e.g., 512 dimensions).

Usage:
- Loaded into device memory at reference load
- Passed to model as conditioning input
- Combined with texture references

---

## 5. Conditioning Mechanism

```
Frame Input (color, depth, motion)
    ↓
Reference Conditioning (concat/attention/AdaIN/embedding)
    ↓
Conditioned Neural Model: "What should THIS character look like?"
    ↓
Conditioned Output
```

---

## 6. Provenance Metadata

```json
{
    "reference_id": "character_103",
    "creator": "Game Publisher Inc.",
    "asset_owner": "Game Publisher Inc.",
    "source": "Licensed Character Asset",
    "license_id": "LIC-2026-10345",
    "permitted_uses": ["runtime_rendering", "neural_reconstruction", "temporal_reconstruction"],
    "prohibited_uses": ["model_training", "standalone_distribution"],
    "attribution_required": true
}
```

NRR doesn't enforce restrictions at binary level. Metadata serves as:
- Contract between creator and user
- Compliance documentation
- Certification/audit input

---

## 7. Reference Loading

```
Game → nrr_reference_load(device, path, &ref)
    ↓
Load .nrrref archive
    ↓
Parse metadata.json
    ↓
Parse provenance.json
    ↓
Load textures to GPU
    ↓
Load identity embedding
    ↓
Reference ready
```

---

## 8. Multiple References Per Frame

```cpp
NRRReferenceSet refs;
refs.facial_reference = face_ref;
refs.hair_reference = hair_ref;
refs.material_reference = material_ref;
refs.skin_reference = skin_ref;

NRR_Render(device, model, &refs, &input, &output);
```

Model uses all provided references for conditioning.

---

## 9. Technical Significance

**Traditional**: "What does a face usually look like?"

**Reference-Conditioned**: "What should THIS developer-authorized character look like?"

Benefits:
- **Control**: Developers specify exact appearance
- **Rights**: Provenance documents asset origin
- **Consistency**: Same character across frames/scenes
- **Pipeline**: Fits existing asset workflows

This separates training data from runtime reference data.

---

*End of Reference-Conditioned Rendering Architecture*
