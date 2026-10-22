# Surrey, BC: Robustness & Crown-Merging Sensitivity Analysis (Comprehensive Audit)

## Methodological Evaluation for Reviewer 2

### 1. Executive Summary & Balanced Scientific Interpretation
Reviewer 2 raised the valid methodological concern that connected-component labeling applied to binary segmentation rasters could fuse adjacent crowns into single large components in dense tree canopy, potentially generating an artifactual negative correlation between income and mean crown area in Surrey, BC ($\rho = -0.14, p = 0.00058$).

To evaluate this concern, we conducted an exhaustive suite of sensitivity analyses across the complete $1,189,424$ individual tree detections in Surrey's $570$ dissemination areas. Our findings provide a nuanced, scientifically balanced conclusion:

1. **Negative Association is Robust to Merging-Artifact Truncations:** Progressively eliminating components exceeding $200\text{ m}^2$, $100\text{ m}^2$, $50\text{ m}^2$, and $25\text{ m}^2$ (the latter removing **$38.7\%$ of Surrey's total canopy area**) yields stable negative correlations ranging between $\rho = -0.137$ and $\rho = -0.148$ ($p \le 0.001$ across all thresholds). In paired bootstrap resampling ($B=2,000$), the 95% confidence intervals for the change in correlation ($\Delta\rho$) tightly overlap zero, demonstrating that large components do not drive the negative income gradient.
2. **Upper-Quantile Consistency:** The 75th percentile ($\rho = -0.138, 95\%\text{ CI: } [-0.217, -0.052], p = 0.00096$) and 90th percentile ($\rho = -0.111, 95\%\text{ CI: } [-0.194, -0.029], p = 0.0077$) of crown area exhibit the same significant negative association as the mean.
3. **Attenuation of Median-Based Association:** When evaluated using median crown area, the correlation attenuates to non-significance ($\rho = -0.036, p = 0.394$). Our audit demonstrates that this attenuation reflects **extreme discrete quantization**: $51.9\%$ of all components in Surrey are 1 or 2 pixels, causing DA-level medians to collapse onto just two discrete integer values (2.0 or 3.0 pixels) in **$82.8\%$ of all dissemination areas** (472 of 570 DAs). This single-pixel floor suppresses variance and weakens rank-order correlations.
4. **Small-Component Filtering Behavior:** Progressively removing small components ($\ge 3\text{ px}, \ge 5\text{ px}, \ge 10\text{ px}$) leaves the correlation negative ($\rho$ between $-0.113$ and $-0.077$). When removing components $<3\text{ px}$ (eliminating $51.9\%$ of components but only $5.9\%$ of canopy area), the correlation remains highly significant ($\rho = -0.1125, p = 0.0072$). When applying heavy filtering ($\ge 10\text{ px}$ or $5\text{--}100\text{ m}^2$, removing $>78\%$ of components), the effect size attenuates to $\rho \approx -0.077$ ($p \approx 0.06$).
5. **Dense Canopy Subsets:** When the top 20% densest canopy neighborhoods are excluded, the negative correlation strengthens to $\rho = -0.176$ ($p = 0.00016$). We note, however, that lower overall canopy cover does not guarantee that individual crowns are free from localized touching.

### 2. Comprehensive Sensitivity Scenarios Table

| Category | Scenario Name | Retained DAs ($N$) | Excluded Comps (%) | Excluded Area (%) | Spearman $\rho$ [95% CI] | $p$-value | $\Delta\rho$ [95% CI] | Sig. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (Manuscript) | Original Published Baseline (m2) | 570 | 0.0% | 0.0% | **-0.1432** [-0.2216, -0.0591] | 6.054e-04 | +0.0000 [0.0000, 0.0000] | *** |
| Large-Component Truncation | Exclude Components > 200 m2 | 570 | 0.1% | 10.8% | **-0.1482** [-0.2263, -0.0639] | 3.859e-04 | -0.0050 [-0.0133, 0.0031] | *** |
| Large-Component Truncation | Exclude Components > 100 m2 | 570 | 0.3% | 17.4% | **-0.1484** [-0.2252, -0.0639] | 3.771e-04 | -0.0052 [-0.0204, 0.0086] | *** |
| Large-Component Truncation | Exclude Components > 50 m2 | 570 | 0.8% | 26.6% | **-0.1412** [-0.2199, -0.0561] | 7.219e-04 | +0.0020 [-0.0246, 0.0258] | *** |
| Large-Component Truncation | Exclude Components > 25 m2 | 570 | 2.0% | 38.7% | **-0.1367** [-0.2161, -0.0516] | 1.066e-03 | +0.0065 [-0.0388, 0.0507] | ** |
| Small-Component Exclusion | Exclude Components < 3 px (~0.97 m2) | 570 | 51.9% | 5.9% | **-0.1125** [-0.1938, -0.0272] | 7.199e-03 | +0.0307 [0.0059, 0.0568] | ** |
| Small-Component Exclusion | Exclude Components < 5 px (~1.62 m2) | 570 | 65.0% | 9.9% | **-0.0941** [-0.1760, -0.0086] | 2.464e-02 | +0.0491 [0.0132, 0.0854] | * |
| Small-Component Exclusion | Exclude Components < 10 px (~3.24 m2) | 570 | 78.5% | 18.0% | **-0.0787** [-0.1622, 0.0073] | 6.057e-02 | +0.0646 [0.0156, 0.1120] | ns |
| Combined Band-Pass Filter | Retain 5 to 100 m2 Components | 568 | 85.7% | 43.0% | **-0.0770** [-0.1601, 0.0093] | 6.666e-02 | +0.0662 [0.0105, 0.1233] | ns |
| Combined Band-Pass Filter | Min 10 px (~3.24 m2) & Max 100 m2 | 570 | 78.8% | 35.4% | **-0.0766** [-0.1594, 0.0077] | 6.762e-02 | +0.0666 [0.0204, 0.1109] | ns |
| Combined Band-Pass Filter | Min 3 px (~0.97 m2) & Max 100 m2 | 570 | 52.2% | 23.4% | **-0.1188** [-0.1991, -0.0335] | 4.524e-03 | +0.0244 [-0.0011, 0.0496] | ** |
| Combined Band-Pass Filter | Min 3 px (~0.97 m2) & Max 50 m2 | 570 | 52.7% | 32.6% | **-0.1092** [-0.1890, -0.0241] | 9.078e-03 | +0.0340 [0.0048, 0.0632] | ** |
| Alternative Quantile Metric | Median Crown Area | 570 | 0.0% | 0.0% | **-0.0357** [-0.1114, 0.0474] | 3.944e-01 | +0.1075 [0.0305, 0.1837] | ns |
| Alternative Quantile Metric | 75th Percentile Crown Area | 570 | 0.0% | 0.0% | **-0.1379** [-0.2168, -0.0515] | 9.611e-04 | +0.0053 [-0.0420, 0.0512] | *** |
| Alternative Quantile Metric | 90th Percentile Crown Area | 570 | 0.0% | 0.0% | **-0.1115** [-0.1943, -0.0287] | 7.718e-03 | +0.0317 [-0.0022, 0.0650] | ** |
| Dense Canopy Exclusion | Exclude Highest 10% Canopy DAs | 513 | 0.0% | 0.0% | **-0.1276** [-0.2130, -0.0384] | 3.800e-03 | +0.0156 [-0.0698, 0.1048] | ** |
| Dense Canopy Exclusion | Exclude Highest 20% Canopy DAs | 456 | 0.0% | 0.0% | **-0.1762** [-0.2634, -0.0861] | 1.560e-04 | -0.0330 [-0.1202, 0.0571] | *** |

### 3. Exact Dataset Reconciliation Audit
- **Component GeoJSON vs. Master Covariates:** Exact 100.0% match across all 573 dissemination areas for both tree count ($1,189,424$ trees) and total pixel area ($13,291,241$ pixels).
- **Dissemination Area Population:** 570 valid DAs in Surrey's primary municipal extent (3 DAs dropped due to zero tree cover, exactly matching the published manuscript baseline).
- **Geodesic Spatial Resolution:** At Surrey's latitude ($49.187^\circ\text{ N}$), true geodesic pixel area is $0.3238\text{ m}^2/\text{pixel}$ (a $29.5\%$ scaling relative to nominal $0.25\text{ m}^2$). All thresholds were audited and confirmed in exact physical units.
