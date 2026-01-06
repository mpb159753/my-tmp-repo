# Tacta Synthetic Data Generation - Development Retrospective

## Overview
This document summarizes the development journey of the synthetic data generation pipeline for the Tacta board game. It highlights the critical challenges encountered, the iterative solutions applied, and the pivotal role of visualization in debugging complex geometric problems.

## Key Challenges & Solutions

### 1. Triangle Orientation (The "Tip" Problem)
**Challenge:** Determining the correct orientation of triangle anchors from raw polygon points was inconsistent. Initial attempts using 90-degree angle detection failed due to noise in manual annotations.
**Solution:** 
- Adopted a robust geometric approach: Simplify polygon to 3 vertices -> Find vertex with largest interior angle -> Define direction from opposite edge midpoint to this vertex.
- **Verification:** Created `debug_triangles.py` to visualize the calculated direction vectors on hundreds of cards, confirming 100% accuracy.

### 2. Coordinate System Mismatch
**Challenge:** During synthesis, generated images showed triangles pointing in incorrect directions (e.g., 105° instead of 255°), even though individual angle calculations were correct.
**Root Cause:** A subtle mismatch between coordinate systems.
- **Mathematical standard:** Counter-Clockwise (CCW) rotation is positive.
- **OpenCV (`getRotationMatrix2D`):** In image coordinates (Y-down), positive angles rotate Counter-Clockwise visually, which corresponds to Clockwise in standard math.
**Solution:** Negated the rotation angle `delta_theta` input to OpenCV.
**Key Lesson:** Logs alone were insufficient ("Target 255, Candidate 105" looked like a logic bug). **Visualization** (drawing vectors on the synthesized output) immediately revealed the systematic rotation error.

### 3. Collision Logic Refinement
**Challenge:** Random placement caused unrealistic overlaps that would be impossible in the physical game.
**Evolution of Rules:**
- *Initial:* Simple bounding box overlap check. (Too loose)
- *Iter 1:* Reject if overlapping >30% of any active anchor. (Better, but allowed cards to slide under others incorrectly)
- *Final:* **"Parent-Child Strictness"**. A new card can **only** overlap its parent anchor. It is rejected if it overlaps **>2%** of any *other* card's area.

## The Value of Visualization "Secondary Checks"

Throughout the process, the introduction of visual debugging tools was the turning point for every major bug fix.

| Stage | Tool | Insight Gained |
|-------|------|----------------|
| **Data Logic** | `visualize_labels.py` | Revealed that "right angle" logic for triangles was flaky on noisy data. |
| **Synthesis** | `debug_synthesis.py` | Overlaid huge ID labels (C0, A1.2) and direction arrows on the final image. |
| **Verification** | **Visual Logs** | Allowed use of "Candidate Angle -> Target Angle" text logs alongside images to pinpoint exact rotation errors. |

**Best Practice:** Do not rely solely on arithmetic logs for geometric problems. Always project the internal state (vectors, centroids, IDs) back onto the image pixels to verify alignment with human intuition.

## Final Synthesis Parameters
- **Canvas:** 2048x2048 (Matches model inference scale)
- **Density:** 5-8 cards per image (Simulates local density of 4K/108-card full table)
- **Strictness:** 2% non-parent overlap tolerance.
