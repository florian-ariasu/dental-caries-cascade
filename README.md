# 🦷 Two-Stage Cascade Framework for Dental Caries Detection

### 📂 Project Structure

#### [M2: Image-Level Baseline](./M2_Baseline_Classifier/)

- **Approach:** Binary classification using ResNet50 on 342 Panoramic (OPG) ROIs.
- **Outcome:** 51.92% Accuracy.
- **Engineering Insight:** Identifed that the model suffered from "Resolution Bottleneck"—it couldn't see incipient caries without localized focus.

#### [M3: Hybrid Cascade System](./M3_Cascade_System/) (Original Contribution)

- **Stage 1 (The Spotter):** YOLOv8 Medium configured for high recall (0.15 threshold) to flag all suspicious regions.
- **Stage 2 (The Expert):** ResNet50 trained on a custom-engineered dataset of 18,000 tooth patches (512x512) with dynamic padding.
- **Hybrid Logic:** A custom inference engine `D(x)` that escalates ambiguous YOLO detections to the ResNet verifier for pixel-level texture analysis.
- **Outcome:** 93.8% mAP@50.

