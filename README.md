# Two-Stage Cascade Framework for Robust Dental Caries Detection

This project investigates deep learning approaches for the automated detection of dental caries in panoramic radiographs (OPG). The research progresses from an image-level classification baseline to a two-stage hybrid cascade architecture designed to improve localization and precision.

---

## Project Structure

### [Milestone 1: Literature Review](./documentation/m1_literature_review.pdf)

- **Objective:** Established the research context by reviewing CNN and Vision Transformer architectures in dental diagnostics.
- **Key Findings:** Identified gaps in clinical generalization, limited multimodal datasets, and the necessity for explainable AI (XAI) tools like Grad-CAM.

### [Milestone 2: Baseline Classifier](./m2-baseline-classifier/)

- **Methodology:** Fine-tuned a **ResNet50** architecture pre-trained on ImageNet for binary classification (Healthy vs. Carious).
- **Dataset:** Utilized 342 panoramic radiographs, applying ROI cropping (512×256) and class-weight tuning to control diagnostic sensitivity.
- **Performance:** Achieved a controlled recall of 0.6471 but limited overall accuracy (0.5192) due to severe class imbalance and lack of spatial localization.

### [Milestone 3: Hybrid Cascade System](./m3-cascade-system/)

- **Architectural Shift:** Reformulated the problem from global classification to a **Localization-Verification Cascade** to address the resolution bottleneck of incipient lesions.
- **Stage 1 — Localization:** **YOLOv8 Medium** acts as a high-recall spotter (threshold 0.15) to propose candidate bounding boxes across the full OPG.
- **Stage 2 — Verification:** **ResNet50** acts as a texture expert, processing high-resolution tooth patches (512×512) to filter false positives from metallic artifacts and overlapping structures.
- **Performance:** Attained a global **mAP50 of 0.938**, with significant precision improvements for both deep and incipient caries.

---

## Methodological Note & Comparison

As noted in the final evaluation, the performance metrics of the M2 Baseline and the M3 Cascade are not directly comparable in a traditional sense.

- **M2** was an image-level classification task on a limited set of 342 ROIs
- **M3** transitioned to a tooth-level detection task, leveraging a significantly expanded dataset of approximately **18,000 tooth patches** generated via automated cropping with dynamic padding

This shift was necessary to capture fine-grained enamel textures that are lost at the global image level. The transition demonstrates that decoupling localization from verification is a more robust approach for clinical radiographs than pure image-level classification.

---

## Documentation

Full technical reports in IEEE format are available in the [documentation](./documentation/) directory:

- **M1:** Related Work & Research Questions
- **M2:** Dataset Collection & Baseline Results
- **M3:** Proposed Methodology & Hybrid Logic
- **M4:** Final Aggregate Paper & Discussion

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
