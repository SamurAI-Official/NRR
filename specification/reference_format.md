# NRR Reference Format Specification

> Defines the .nrrref file format and reference asset structure

---

## 1. Overview

NRR references are developer-supplied assets that condition the neural renderer toward specific visual identities. They are distributed as `.nrrref` files containing:
- Reference textures
- Identity embeddings
- Provenance metadata

---

## 2. File Structure

```
.nrrref (ZIP archive)
├── metadata.json              # Reference metadata
├── textures/                  # Reference textures
│   ├── face.dds              # Facial reference
│   ├── hair.dds              # Hair reference
│   ├── skin.dds              # Skin reference
│   ├── clothing.dds          # Clothing reference
│   ├── material.dds          # Material reference
│   └── expression.dds        # Expression reference
├── embeddings/                # Learned embeddings (optional)
│   └── identity_embedding.npz
└── provenance.json            # Provenance metadata (can be in metadata.json)
```

---

## 3. Reference Types

### 3.1 Reference Categories

| Category | Purpose | Typical Content |
|----------|---------|-----------------|
| `facial_reference` | Character face conditioning | High-res face render |
| `material_reference` | Material appearance conditioning | Material sample |
| `hair_reference` | Hair appearance conditioning | Hair render/sample |
| `skin_reference` | Skin appearance conditioning | Skin texture/sample |
| `clothing_reference` | Clothing conditioning | Clothing render/sample |
| `expression_reference` | Expression conditioning | Expressions atlas |
| `identity_embedding` | Learned identity vector | Neural embedding |

### 3.2 Reference Presence

Not all references must be present. The model declares which references it requires in its metadata. Optional references can be omitted.

---

## 4. Reference Texture Specs

### 4.1 Format Requirements

- Must be in a GPU-readable format (DDS, PNG, or backend-specific format)
- Resolution should match the model's expected reference resolution
- Color space should be specified in metadata

### 4.2 Texture Metadata

```json
{
    "name": "facial_reference",
    "resolution": [1024, 1024],
    "format": "RGB8",
    "color_space": "sRGB",
    "description": "Character face reference at neutral expression"
}
```

---

## 5. Identity Embedding

### 5.1 Structure

An identity embedding is a learned vector that captures the essence of a visual identity.

```json
{
    "name": "identity_embedding",
    "dimensions": 512,
    "format": "FLOAT32",
    "description": "Identity embedding vector for character 103",
    "data_file": "embeddings/identity_embedding.npz"
}
```

### 5.2 Usage

- Loaded into device memory at reference load time
- Passed to the neural model as part of the reference set
- Can be combined with texture references for richer conditioning

---

## 6. Provenance Metadata

### 6.1 Structure

```json
{
    "reference_id": "character_103",
    "version": "1.0.0",

    "creator": {
        "name": "Game Publisher Inc.",
        "contact": "licensing@publisher.example.com"
    },

    "asset_owner": "Game Publisher Inc.",
    "source": "Licensed Character Asset",
    "source_asset_id": "CHAR-103-REF-001",

    "creation_date": "2026-09-01",
    "expiration_date": null,

    "license": {
        "id": "LIC-2026-10345",
        "type": "runtime_rendering",
        "version": "1.0"
    },

    "permitted_uses": [
        "runtime_rendering",
        "neural_reconstruction",
        "temporal_reconstruction",
        "upscaling"
    ],

    "prohibited_uses": [
        "model_training",
        "standalone_distribution",
        "modification_and_redistribution"
    ],

    "attribution_required": true,
    "attribution_text": "Character design by Game Publisher Inc.",

    "chain_of_custody": [
        {
            "date": "2026-09-01",
            "action": "creation",
            "entity": "Original Creator"
        },
        {
            "date": "2026-09-05",
            "action": "licensing",
            "entity": "Game Publisher Inc."
        }
    ],

    "integrity": {
        "sha256": "a1b2c3d4e5f6...",
        "format": "sha256"
    }
}
```

---

## 7. Usage Permissions

### 7.1 Permission Levels

| Permission | Description |
|------------|-------------|
| `runtime_rendering` | Can be used during runtime rendering |
| `neural_reconstruction` | Can be used as neural reconstruction input |
| `temporal_reconstruction` | Can be used in temporal processing |
| `upscaling` | Can be used for upscaling reference |
| `model_training` | Can be used to train new models (rarely granted) |

### 7.2 Prohibition Enforcement

NRR does not enforce usage restrictions at the binary level. The provenance metadata serves as:
- A contract between asset creator and user
- Documentation for compliance verification
- Input to certification and audit systems

---

## 8. Reference Loading

### 8.1 Loading Process

1. Game calls `nrr_reference_load(device, path, &reference)`
2. NRR loads and validates the `.nrrref` file
3. Resources are allocated on the device
4. Reference is ready for use in rendering

### 8.2 Reference Lifecycle

- References are created once and used for many frames
- References can be updated (reloaded) if assets change
- References are destroyed when no longer needed

---

## 9. Multiple References

A single model invocation can receive multiple references:

```cpp
NRRReferenceSet references;

references.identity = character_face_reference;
references.hair = hair_reference;
references.material = material_reference;
references.skin = skin_reference;
```

The model uses all provided references to condition its output.

---

*End of Reference Format Specification*
