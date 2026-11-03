# Scientific Interpretation of Austin HOLC Redlining Figures

**Companion Document to:** Figures 1, 2, and 3 (`results/austin_redlining/`)  
**Audited Datasets:** `austin_holc_polygon_metrics.csv`, `austin_holc_grade_metrics.csv`, `austin_holc_sensitivity_analysis.csv`

---

## 1. Executive Summary

This document provides a concise scientific interpretation of the three figures generated for the exploratory analysis of historical Home Owners' Loan Corporation (HOLC) residential security grades (1937) and contemporary high-resolution urban tree distributions in Austin, Texas.

The overarching conclusion is that **while an exploratory visual comparison reveals striking spatial patterns, the apparent "Grade A canopy advantage" is mathematically fragile and entirely contingent upon a single dominant polygon (Polygon 9093).** Furthermore, severe sample size constraints ($N_{\text{total}} = 18$; $N_C = 1$) and geographic clustering preclude inferential statistical modeling or causal claims.

---

## 2. Interpretation of Figure 1: Spatial Distribution and Study Footprint

**Artifacts:** `figure_1_holc_austin_tree_distribution.png` & `.pdf`

### 2.1 Key Visual Findings
1. **Historical Subsetting of Contemporary Austin (Panel a):**  
   The municipal boundary of contemporary Austin spans $861.92\text{ km}^2$ (with $828.64\text{ km}^2$ of Census land area). The 18 historical HOLC polygons encompass only $45.68\text{ km}^2$, or **$5.30\%$ of the modern city**. Over $94\%$ of contemporary Austin lies outside the historical survey area. Cartographically, the HOLC survey is confined to the early-20th-century urban core.
2. **The Un-Graded Core (Panel b):**  
   A contiguous $24.78\text{ km}^2$ central area was intentionally left un-graded by the HOLC in 1937 because it comprised commercial, financial, and institutional properties rather than residential mortgage assets. This un-graded zone encompasses Downtown Austin, the Texas State Capitol complex, the University of Texas at Austin main campus, and the Colorado River corridor.
3. **East-West Geographic Sorting (Panel b):**  
   Historical grades exhibit stark geographic segregation:
   - **Grade A (Green, $n=3$):** Concentrated in the elevated western limestone hills (Old West Austin, Pemberton Heights, Enfield) bordering Lake Austin.
   - **Grade B (Blue, $n=7$):** Located primarily in north-central residential subdivisions (Hyde Park) and south-central neighborhoods (Travis Heights).
   - **Grade C (Yellow, $n=1$):** A single polygon located south of the river (Bouldin Creek / South Congress area).
   - **Grade D (Red, $n=7$):** Concentrated east of modern Interstate 35 (the historical East Avenue line established by the 1928 Austin City Plan) and along the Missouri-Kansas-Texas (MKT) and Southern Pacific railroad corridors, as well as industrial pockets in South Austin.
4. **Spatial Footprint of Polygon 9093 (Panel b):**  
   Polygon 9093 occupies nearly the entire western residential quadrant ($12.88\text{ km}^2$), abutting the Colorado River and possessing dense, mature canopy cover.

---

## 3. Interpretation of Figure 2: Polygon Metrics Across Historical Grades

**Artifacts:** `figure_2_polygon_tree_metrics_by_grade.png` & `.pdf`

### 3.1 Key Quantitative Patterns
Figure 2 plots all 18 historical polygons as individual observations alongside area-weighted grade-level aggregates (diamonds and crossbars), avoiding misleading boxplots or unweighted polygon averages:

1. **Panel (a): Tree Density ($\text{trees/km}^2$)**
   - **Grade A:** Weighted density is $1,477.5\text{ trees/km}^2$. However, individual polygon density varies twofold, ranging from $726.3\text{ trees/km}^2$ in Polygon 9083 to $1,633.4\text{ trees/km}^2$ in Polygon 9093.
   - **Grade B:** Weighted density is $849.0\text{ trees/km}^2$, with individual polygons spanning $417.6$ to $1,231.4\text{ trees/km}^2$.
   - **Grade C:** Represented by a single observation ($N_C = 1$, Polygon 9084) at $629.2\text{ trees/km}^2$. Within-grade variance is non-existent.
   - **Grade D:** Weighted density is $700.1\text{ trees/km}^2$, with individual polygons spanning $597.7$ to $1,183.5\text{ trees/km}^2$.
   - **Critical Observation:** Grade D tree density ($700.1\text{ trees/km}^2$) is higher than Grade C ($629.2\text{ trees/km}^2$), directly violating the assumption of a monotonic decline across grades. Furthermore, several Grade D polygons (e.g., Polygon 9096 at $1,183.5\text{ trees/km}^2$) exhibit higher tree densities than the Grade B average.

2. **Panel (b): Mean Detected Crown Area ($\text{m}^2/\text{tree}$)**
   - Individual tree crown sizes show remarkable stability across historical grades, averaging between $1.34\text{ m}^2$ and $1.65\text{ m}^2$ per tree.
   - Trees in Grade A have a slightly *smaller* average detected crown area ($1.343\text{ m}^2$) than trees in Grade B ($1.654\text{ m}^2$), Grade C ($1.517\text{ m}^2$), or Grade D ($1.608\text{ m}^2$).
   - This occurs because Polygon 9093 contains thousands of densely packed small-to-medium residential yard trees and understory vegetation that segment into smaller discrete crown polygons ($1.317\text{ m}^2/\text{tree}$).

3. **Panel (c): Summed Detected Crown-Area Fraction (%)**
   - Weighted crown-area fraction shows Grade A ($0.198\%$) followed by Grade B ($0.140\%$), Grade D ($0.113\%$), and Grade C ($0.095\%$).
   - Again, Grade D exhibits a higher crown-area fraction than Grade C, refuting a simple linear gradient.
   - The citywide mean for this metric across all 1,902,398 trees is $0.1736\%$. Thus, only Grade A exceeds the contemporary citywide average canopy fraction; Grades B, C, and D are all below the citywide average.

---

## 4. Interpretation of Figure 3: Influence and Sensitivity Analysis

**Artifacts:** `figure_3_influence_sensitivity_analysis.png` & `.pdf`

### 4.1 Structural Dominance of Polygon 9093 (Panel a)
Panel (a) quantifies the extraordinary concentration of Grade A within Polygon 9093:
- **Area Share:** Polygon 9093 comprises **$81.46\%$** ($12.88\text{ km}^2$) of all Grade A land area ($15.81\text{ km}^2$).
- **Tree Share:** Polygon 9093 contains **$90.05\%$** (21,034 trees) of all Grade A trees ($23,357$).
- The remaining two Grade A polygons (9090 and 9083) represent only $18.54\%$ of area and $9.95\%$ of trees.

### 4.2 Collapse of the Grade A Advantage Under Exclusion (Panels b & c)
Panels (b) and (c) compare full-sample descriptive results against the exclusion of Polygon 9093, using Grade B as an authoritative baseline across equal-sized dedicated panels:
- **Tree Density (Panel b):** When Polygon 9093 is excluded, Grade A tree density **collapses from $1,477.5\text{ trees/km}^2$ down to $792.5\text{ trees/km}^2$**, a reduction of $-46.4\%$. In this exclusion scenario, Grade A density falls **$6.6\%$ below Grade B** ($849.0\text{ trees/km}^2$).
- **Summed Crown-Area Fraction (Panel c):** When Polygon 9093 is excluded, Grade A crown fraction **drops from $0.198\%$ down to $0.125\%$**, a reduction of $-36.9\%$. In this exclusion scenario, Grade A canopy fraction falls **$10.8\%$ below Grade B** ($0.140\%$).
- **Conclusion:** The apparent environmental "superiority" of Grade A in Austin is entirely driven by a single massive polygon in West Austin. The remaining Grade A polygons (9090 and 9083) have tree metrics that are indistinguishable from or lower than Grade B.

### 4.3 Systematic Jackknife Sensitivity Across All Polygons (Panel d)
Panel (d) displays the Leave-One-Polygon-Out (LOPO) percentage shifts for all 18 historical polygons:
- **Polygon 9093 (Grade A):** Induces a massive **$-46.4\%$ shift** in density and **$-36.9\%$ shift** in crown fraction.
- **Polygon 9092 (Grade B):** Induces a **$-14.1\%$ shift** in Grade B density (as it represents $8.85\text{ km}^2$ of Grade B's $14.03\text{ km}^2$).
- **Polygon 9086 (Grade D):** Induces a **$+14.4\%$ shift** in Grade D density (excluding this large industrial/rail polygon increases the density of the remaining residential Grade D enclaves).
- **Other 14 Polygons:** Exert negligible leverage (absolute parameter shifts $< 5\%$).
- **Grade C:** Mathematically undefined ($N_C = 1$), explicitly documented with a sensitivity callout footnote.

---

## 5. Summary: Robust vs. Sensitive Findings

```
+---------------------------------------------------------------------------------------------------------+
| Synthesis of Analytical Robustness                                                                      |
+----------------------------------------------------+----------------------------------------------------+
| Robust Findings (Data-Supported)                   | Sensitive / Fragile Patterns (Do Not Over-Interpret)|
+----------------------------------------------------+----------------------------------------------------+
| 1. Geographic Extent: HOLC coverage is strictly a   | 1. "Grade A Canopy Premium": Completely dependent  |
|    historical subset (5.30%) of contemporary       |    on Polygon 9093; inverts below Grade B when     |
|    Austin, separated by an un-graded core.         |    Polygon 9093 is excluded.                       |
|                                                    |                                                    |
| 2. East-West Topographic Sorting: Higher tree      | 2. Monotonic Grading Gradient: Refuted by data;    |
|    canopy in western limestone hills vs. lower     |    Grade D has higher density (700.1 vs 629.2) and |
|    canopy in eastern flatlands & rail corridors.   |    canopy cover (0.113% vs 0.095%) than Grade C.  |
|                                                    |                                                    |
| 3. Stable Individual Tree Size: Mean detected      | 3. Grade C Generalizability: Completely invalid    |
|    crown areas are remarkably uniform (1.3–1.7 m²)  |    for comparative inference (N = 1 polygon;       |
|    across all grades.                              |    within-grade variance is undefined).            |
|                                                    |                                                    |
| 4. Small Area Over-Weighting: Naïve unweighted     | 4. Causal Redlining Effect: Unsubstantiated;       |
|    polygon averages distort grade values by up to  |    patterns predate 1937 via the 1928 Austin       |
|    28% compared to area-weighted aggregates.       |    Master Plan and natural topography.             |
+----------------------------------------------------+----------------------------------------------------+
```
