# Medical Imaging Fundamentals

> This document covers the core concepts you need to understand for medical imaging AI.
> Essential knowledge for technical interviews at companies like Palantir, OpenAI, or healthcare AI startups.

## Table of Contents
1. [DICOM Format](#dicom-format)
2. [CT Imaging Physics](#ct-imaging-physics)
3. [Hounsfield Units](#hounsfield-units)
4. [Windowing](#windowing)
5. [3D Image Representation](#3d-image-representation)
6. [Why Medical Images Are Different](#why-medical-images-are-different)

---

## DICOM Format

### What is DICOM?

**DICOM** (Digital Imaging and Communications in Medicine) is the universal standard for medical images. Think of it as "the JPEG of healthcare" - but with much more complexity.

### Why Not Use PNG/JPEG?

| Feature | JPEG/PNG | DICOM |
|---------|----------|-------|
| Bit depth | 8-bit (256 levels) | 12-16 bit (4,096-65,536 levels) |
| Metadata | Basic EXIF | Hundreds of standardized tags |
| Physical units | Pixels | Real-world millimeters |
| 3D support | None | Native |
| Clinical validation | None | FDA-cleared workflows |

### Key DICOM Tags

```python
# Tags you'll work with frequently
important_tags = {
    'PatientID': 'Unique patient identifier',
    'Modality': 'CT, MR, US, etc.',
    'PixelSpacing': 'Physical size of each pixel [row, col] in mm',
    'SliceThickness': 'Distance between slice centers in mm',
    'ImagePositionPatient': '3D coordinates of first pixel',
    'RescaleSlope': 'For HU conversion',
    'RescaleIntercept': 'For HU conversion',
    'WindowCenter': 'Default display setting',
    'WindowWidth': 'Default display setting',
}
```

---

## CT Imaging Physics

### How CT Scanners Work

1. **X-ray Source** rotates around patient
2. **Detectors** measure X-ray attenuation (how much was absorbed)
3. **Computer** reconstructs cross-sectional images from projections
4. **Stack** multiple slices to create 3D volume

```
    X-ray Source
         |
         v
    +---------+
    | Patient |  ← X-rays pass through
    +---------+
         |
         v
    Detector Array (measures remaining X-rays)
```

### CT Angiography (CTA)

For detecting aneurysms, we use **CT Angiography**:
- Patient receives IV contrast agent (iodine-based)
- Contrast makes blood vessels very bright on CT
- Timing is critical - scan must happen when contrast reaches brain vessels
- This is why aneurysms are visible: the contrast-filled vessel outpouching

---

## Hounsfield Units

### The Physical Scale of CT

Unlike arbitrary pixel values, CT values have **physical meaning**:

```
HU Scale:
-1000 HU ─┬─ Air
          │
  -500 HU ─┤─ Lung tissue
          │
     0 HU ─┼─ Water (by definition)
          │
   +40 HU ─┤─ Soft tissue, gray matter
          │
  +100 HU ─┤─ Blood (with contrast)
          │
  +400 HU ─┤─ Bone (trabecular)
          │
+1000 HU ─┴─ Dense bone
          │
+3000 HU ──── Metal implants
```

### Conversion Formula

Raw DICOM pixel values are NOT Hounsfield Units. You must convert:

```python
HU = pixel_value * RescaleSlope + RescaleIntercept
```

**Interview Question**: "Why can't we just normalize CT images to [0,1]?"

**Answer**: HU values encode physical tissue properties. Normalizing without windowing first would:
1. Give different results for different scanners (different value ranges)
2. Compress the diagnostic range (vessels vs. soft tissue barely distinguishable)
3. Lose the ability to identify tissue types by their HU range

---

## Windowing

### The Problem

CT values span ~4000 HU (-1000 to +3000), but:
- Monitors display ~256 gray levels
- Human eyes distinguish ~100 gray levels
- We need to focus on relevant tissue range

### The Solution: Windowing

**Window** = range of HU values mapped to display gray levels

```
Example: CTA Window for Vessels
Window Center: 150 HU
Window Width: 400 HU

          -1000          0    150         400        1000
             |           |     |           |           |
HU scale:  ████████████░░░░░▓▓▓▓▓▓▓▓▓▓░░░░░████████████
                          |___________|
                           Visible range
                           (mapped to 0-255)

-50 HU and below → Black (0)
350 HU and above → White (255)
-50 to 350 HU → Linear gradient
```

### Common Windows

| Window Name | Center | Width | Use Case |
|-------------|--------|-------|----------|
| Brain | 40 | 80 | Gray/white matter |
| Stroke | 32 | 8 | Subtle hemorrhage |
| CTA | 150 | 400 | Blood vessels |
| Bone | 600 | 2800 | Skull, spine |
| Lung | -600 | 1600 | Lung parenchyma |

### Code Example

```python
def apply_window(volume_hu, center, width):
    """
    Apply CT windowing.
    
    Args:
        volume_hu: Volume in Hounsfield Units
        center: Window center (level)
        width: Window width
    
    Returns:
        Windowed volume normalized to [0, 1]
    """
    min_hu = center - width / 2
    max_hu = center + width / 2
    
    # Clip and normalize
    windowed = np.clip(volume_hu, min_hu, max_hu)
    normalized = (windowed - min_hu) / (max_hu - min_hu)
    
    return normalized
```

---

## 3D Image Representation

### From 2D Slices to 3D Volume

CT scanners produce individual 2D slices that we stack into 3D:

```
Slice 1    Slice 2    Slice 3       3D Volume
+----+    +----+    +----+          +--------+
|    |    |    |    |    |    →    /        /|
|    |    |    |    |    |        /        / |
+----+    +----+    +----+       +--------+  |
                                 |        |  +
                                 |        | /
                                 +--------+

Shape: (num_slices, height, width) = (D, H, W)
```

### Voxels (3D Pixels)

In 3D, pixels become **voxels** (volume elements):
- Each voxel has a physical size in mm
- **Isotropic**: same size in all dimensions (1×1×1 mm)
- **Anisotropic**: different sizes (0.5×0.5×2.0 mm typical for CT)

### Why Spacing Matters

```python
# Scanner A: 0.5 × 0.5 × 1.0 mm spacing
# A 3×3×3 kernel covers: 1.5 × 1.5 × 3.0 mm physical space

# Scanner B: 1.0 × 1.0 × 3.0 mm spacing  
# A 3×3×3 kernel covers: 3.0 × 3.0 × 9.0 mm physical space

# Problem: Same kernel, VERY different physical coverage!
# Solution: Resample to consistent spacing before training
```

---

## Why Medical Images Are Different

### Key Differences from Natural Images

| Aspect | Natural Images | Medical Images |
|--------|---------------|----------------|
| Dimensions | 2D (H×W×3) | 3D (D×H×W×1) |
| Values | 0-255 arbitrary | Physical units (HU) |
| Scale | Pixels | Millimeters |
| Resolution | Fixed | Variable per scan |
| Size | ~1 MB | ~100-500 MB |
| Annotations | Bounding boxes | Often sparse/none |

### Implications for Deep Learning

1. **Memory**: 3D volumes are huge. A 512×512×300 float32 volume = 300 MB
2. **Architecture**: Need 3D convolutions, not 2D
3. **Preprocessing**: Must handle variable spacing
4. **Augmentation**: Need 3D-aware transforms
5. **Evaluation**: Clinical metrics matter (sensitivity for cancer detection)

### Interview Discussion Points

**Q: Why use 3D CNNs instead of processing slices with 2D CNNs?**

A: Brain aneurysms are 3D structures. A 2D CNN only sees one slice at a time - it can't understand that a bright spot in slice 50 is connected to bright spots in slices 48-52 (forming a sphere-like aneurysm). 3D convolutions capture these volumetric relationships.

**Q: What's the biggest challenge in medical imaging AI?**

A: Data! Medical images are expensive to acquire, label (need expert radiologists), and share (privacy regulations). Models trained on one hospital's scanner may not work on another's. Preprocessing and augmentation are critical for generalization.

---

## Summary Checklist

Before your interview, make sure you can explain:

- [ ] Why DICOM exists and what information it contains
- [ ] What Hounsfield Units are and why they matter
- [ ] How windowing works and why we need it
- [ ] Why medical images need resampling
- [ ] The difference between isotropic and anisotropic voxels
- [ ] Why 3D CNNs are necessary for volumetric analysis
- [ ] The challenges unique to medical imaging AI

---

*Next: [02_3d_cnn_architecture.md](02_3d_cnn_architecture.md) - Understanding 3D Convolutional Neural Networks*

