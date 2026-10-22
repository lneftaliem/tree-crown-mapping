# Quantitative Methods: Historical HOLC Redlining and Contemporary Urban Tree Canopy Analysis in Austin, Texas

**Document Version:** 1.0 (Audited)  
**Study Location:** Austin, Texas, USA  
**Code:** `generate_redlining_figures.py`; outputs in `results/austin_redlining/`  
**Authoritative Source Datasets:**
- Tree Canopy Points: `austin_ripley_points_with_income_corrected.geojson` ($N = 1,902,398$)
- Historical HOLC Boundaries: `austin_redlining_data.json` ($N_{\text{polygons}} = 18$)
- Citywide Spatial Footprint: `all_cities_urban_covariates_corrected.csv` (Austin Census land area: $828.64\text{ km}^2$; Municipal boundary: $861.92\text{ km}^2$)

---

## 1. Historical HOLC Geospatial Data Provenance and Classification

### 1.1 Historical Context and Survey Origin
Historical residential mortgage risk maps were acquired from the *Mapping Inequality: Redlining in New Deal America* project, hosted by the Digital Scholarship Lab (DSL) at the University of Richmond (Nelson et al., 2023; DSL City ID: 246). The underlying survey for Austin, Texas, was conducted by the Home Owners' Loan Corporation (HOLC) Division of Research and Statistics in collaboration with local real estate brokers, mortgage lenders, and appraisers, with the survey and descriptive area descriptions completed on **June 30, 1937**.

The HOLC program was created under the Home Owners' Loan Act of 1933 to refinance distressed mortgages during the Great Depression. The resulting "Residential Security Maps" appraised neighborhood lending risk using four standardized grades:
- **Grade A ("First Grade", Green):** Described by HOLC as "Best"; characterized as homogenous, affluent, recently developed residential areas occupied by business and professional executives, free of "undesirable racial or foreign elements."
- **Grade B ("Second Grade", Blue):** Described as "Still Desirable"; settled neighborhoods that were completely developed but stable, retaining good mortgage desirability despite moderate building age.
- **Grade C ("Third Grade", Yellow):** Described as "Definitely Declining"; neighborhoods characterized by aging housing stock, lack of deed restrictions, and the infiltration of lower-income residents or non-white populations.
- **Grade D ("Fourth Grade", Red):** Described as "Hazardous"; designated for denial or strict restriction of mortgage lending, explicitly characterized by low-income residents, older dilapidated structures, proximity to industrial contamination, and heavy concentrations of African American or Mexican American residents.

### 1.2 Preservation of Historical Analytical Boundaries
Crucially, these historical grades reflect **1937 institutional underwriting evaluations**, not present-day neighborhood boundaries, legal zoning categories, or contemporary socioeconomic status. In accordance with rigorous historical-geographic methodology, grades are analyzed strictly as historical spatial designations without projecting contemporary demographic or administrative definitions backward in time.

### 1.3 Geographic Extent and Core Institutional Exclusions
The digital HOLC GeoJSON for Austin comprises **18 discrete residential polygons** encompassing a combined area of **$45.68\text{ km}^2$** ($45,684,690\text{ m}^2$). This historical survey area corresponds to:
- **$5.30\%$** of the contemporary municipal boundary ($861.92\text{ km}^2$, encompassing municipal water surfaces and extraterritorial annexations).
- **$5.51\%$** of the U.S. Census Bureau (2020 TIGER/Line) Austin land-area footprint ($828.64\text{ km}^2$).

HOLC surveyed only developed residential neighborhoods. Consequently, a contiguous **$24.78\text{ km}^2$** central urban core (comprising the Texas State Capitol, Downtown commercial and financial district, the University of Texas at Austin main campus, and the Colorado River (Lady Bird Lake) corridor) was deliberately left **un-graded** in 1937. All un-graded areas are preserved as an explicit, neutral geographic background layer in all spatial representations.

```
+---------------------------------------------------------------------------------------+
| Austin HOLC Spatial Sample Breakdown (June 30, 1937)                                  |
+-------+-------------------------+---------------+---------------+---------------------+
| Grade | Historical Description  | Polygon Count | Total Area    | Share of HOLC Area  |
+-------+-------------------------+---------------+---------------+---------------------+
| A     | "First Grade" / Best    | 3             | 15.81 km²     | 34.60%              |
| B     | "Second Grade"          | 7             | 14.03 km²     | 30.72%              |
| C     | "Third Grade"           | 1             |  3.88 km²     |  8.48%              |
| D     | "Fourth Grade"          | 7             | 11.97 km²     | 26.19%              |
+-------+-------------------------+---------------+---------------+---------------------+
| Total | Complete HOLC Footprint | 18            | 45.68 km²     | 100.00%             |
+-------+-------------------------+---------------+---------------+---------------------+
```

---

## 2. Contemporary High-Resolution Tree Canopy Point Data

### 2.1 Provenance and Feature Definition
Contemporary tree distributions were derived from the authoritative corrected point layer `austin_ripley_points_with_income_corrected.geojson`, containing **1,902,398 detected tree features** spanning the Austin municipal study extent.

Each point in the dataset represents a single detected tree crown centroid derived from high-resolution satellite imagery:
- **Sensor Platform:** Maxar WorldView-2 / WorldView-3 satellite constellations.
- **Ground Sample Distance (GSD):** $0.50\text{ m}$ pan-sharpened multispectral ground resolution.
- **Extraction Architecture:** Deep convolutional neural network (U-Net) trained for individual tree crown segmentation across urban canopies.
- **Centroid Reduction:** Continuous polygonal crown segmentations were reduced to geographic point coordinates $(x_i, y_i)$ at the crown polygon centroid, with geometric crown properties transferred as vector attributes.

### 2.2 Crown-Area Attribute Derivation and Quantization
Each point feature contains a scalar attribute `crown_area_m2`, which records the planimetric horizontal crown area in square meters. In the original image-processing pipeline, crown area was computed by counting contiguous segmented image pixels and multiplying by the projected ground area of an individual raster pixel:

$$A_{\text{pixel}} = \Delta x \times \Delta y = 0.5048\text{ m} \times 0.5053\text{ m} = 0.255065\text{ m}^2$$

Consequently, the recorded crown areas represent quantized integer multiples of $A_{\text{pixel}}$:
- $1\text{ pixel} = 0.255\text{ m}^2$ (minimum detectable canopy unit)
- $2\text{ pixels} = 0.510\text{ m}^2$
- $3\text{ pixels} = 0.765\text{ m}^2$ (empirical median across Austin)
- $4\text{ pixels} = 1.020\text{ m}^2$
- $5\text{ pixels} = 1.275\text{ m}^2$

Quality control audit confirmed:
- Zero records with missing (`NULL` or `NaN`), zero, or negative crown areas.
- Tree count within the 18 HOLC polygons: **$N_{\text{trees}} = 46,088$**.
- Mean detected crown area across all HOLC trees: **$1.488\text{ m}^2$** (corresponding to $5.83$ pixels).
- Citywide mean detected crown area: **$1.5001\text{ m}^2$** ($5.88$ pixels).

### 2.3 Critical Distinction: Summed Detected Crown Area vs. Continuous Raster Canopy Cover
**Methodological Requirement:** Summed detected crown-area fraction is strictly defined as the spatial summation of discrete crown-area attributes associated with point features within a given geographic boundary, divided by the total ground area of that boundary. 

$$\text{Summed Crown Fraction} = \frac{\sum_{i=1}^{N} \text{crown\_area\_m2}_i}{\text{Ground Area (m}^2\text{)}} \times 100\%$$

This metric **must not be confused with or reported as continuous raster canopy cover**. It does not measure continuous leaf area index (LAI), fractional vegetation coverage from NDVI, or the planar geometric union of overlapping polygon crowns. In dense forest stands or overlapping canopies, the summation of individual crown areas can overestimate ground projection if crowns overlap, but here it serves as an audited, point-associated proxy for detected cumulative crown area.

---

## 3. Coordinate Reference Systems (CRS) and Geometric Transformations

To eliminate area distortion while maintaining cartographic consistency with existing manuscript figures, two distinct coordinate reference systems are deployed:

### 3.1 Geographic Visualization CRS: WGS 84 (`EPSG:4326`)
All cartographic maps and spatial boundary displays are rendered in **WGS 84 (`EPSG:4326`)** unprojected latitude/longitude coordinates (degrees). This aligns directly with:
- The GeoJSON geographic standard specifications.
- The visualization conventions used in `city_tree_analysis.py` for manuscript chloropleth figures.

### 3.2 Projected Equal-Area / Conformal Calculation CRS: UTM Zone 14N (`EPSG:32614`)
All spatial overlays, distance thresholds, spatial indexing, and planimetric polygon area calculations are executed in **Universal Transverse Mercator (UTM) Zone 14N (`EPSG:32614`)**, referenced to the WGS 84 datum with units of international meters:
- **Central Meridian:** $99^\circ\text{W}$ (Austin is located at approximately $97.74^\circ\text{W}$, within $1.3^\circ$ of the central meridian).
- **Scale Factor:** $k_0 = 0.9996$, resulting in a local scale distortion of less than $0.05\%$.
- **Audit Verification:** Prior scratch scripts in early exploratory testing inadvertently projected to Texas Central State Plane (`EPSG:2277`, US Survey Feet) and divided by $10^6$ instead of $5,280^2$. A rigorous QC audit verified that **no State Plane or square-foot values entered the final analytical datasets**. All areas in square kilometers are calculated strictly as $\text{Area (m}^2\text{)} / 1,000,000$ in `EPSG:32614`.

---

## 4. Spatial Assignment and Point-in-Polygon Topology

### 4.1 Spatial Predicate
Tree points were matched to historical HOLC polygons using a rigorous planar point-in-polygon spatial join (`predicate='within'` in `geopandas.sjoin`, utilizing GEOS planar topological indexing):

$$P_i \in \text{Polygon}_j \iff \text{ST\_Within}(P_i, \text{Polygon}_j) = \text{True}$$

### 4.2 Boundary Overlap and Duplicate Verification
Because the 18 HOLC polygons are mutually disjoint ($\text{Polygon}_j \cap \text{Polygon}_k = \emptyset, \forall j \neq k$), no single point could intersect multiple polygons. An audit of index uniqueness confirmed:
- Zero tree points were assigned to multiple polygons.
- Points falling exactly on polygon boundary coordinates follow the standard half-open boundary topological rule implemented in GEOS.
- Because tree locations are represented as point centroids, crown edge overhang beyond polygon boundaries is not partitioned or clipped, matching the discrete object methodology applied across all other census tracts and block groups in the manuscript.

---

## 5. Mathematical Outcome Definitions

For each individual HOLC polygon $j \in \{1, 2, \dots, 18\}$, three primary scientific outcomes are computed:

### 5.1 Individual-Polygon Tree Density ($D_j$)
Tree density expresses the abundance of detected trees per unit area, scaled to square kilometers:

$$D_j = \frac{N_j}{A_j} \quad \left[\text{trees / km}^2\right]$$

where $N_j$ is the total count of tree centroids located within polygon $j$, and $A_j$ is the planimetric ground area of polygon $j$ in square kilometers ($A_j = \text{Area}_{\text{m}^2, j} \times 10^{-6}$).

### 5.2 Individual-Polygon Mean Detected Crown Area ($\bar{C}_j$)
Mean detected crown area measures the average planimetric horizontal crown size of trees within polygon $j$:

$$\bar{C}_j = \frac{1}{N_j} \sum_{i=1}^{N_j} C_{ij} \quad \left[\text{m}^2 / \text{tree}\right]$$

where $C_{ij}$ represents the scalar attribute `crown_area_m2` for tree $i$ in polygon $j$.

### 5.3 Individual-Polygon Summed Detected Crown-Area Fraction ($F_j$)
Summed crown-area fraction expresses the proportion of polygon ground area covered by the cumulative crown area of detected trees:

$$F_j = \frac{\sum_{i=1}^{N_j} C_{ij}}{\text{Area}_{\text{m}^2, j}} \times 100\% = \frac{\sum_{i=1}^{N_j} C_{ij}}{A_j \times 10^6} \times 100\% \quad [\%]$$

### 5.4 Algebraic Equivalence
By algebraic substitution, the three metrics satisfy the direct identity:

$$F_j = D_j \times \bar{C}_j \times 10^{-4}\%$$

For example, for Polygon 9093:
$$D_{9093} = 1,633.39\text{ trees/km}^2, \quad \bar{C}_{9093} = 1.3168\text{ m}^2$$
$$F_{9093} = 1,633.3865 \times 1.316824 \times 10^{-4} = 0.215088\%$$

---

## 6. Grade-Level Aggregation and Weighting Schemes

A primary methodological hazard in spatial redlining analysis is the **ecological fallacy of unweighted polygon averaging**. When polygons within a grade differ by more than two orders of magnitude in spatial area (e.g., in Grade D, Polygon 9086 is $6.95\text{ km}^2$ while Polygon 9097 is $0.09\text{ km}^2$), computing an unweighted mean of polygon-level rates gives equal statistical weight to an area that is 75 times smaller, severely biasing grade-level comparisons.

We explicitly distinguish between polygon-level observations and **area-weighted grade-level aggregates**.

### 6.1 Area-Weighted Grade Tree Density ($D_g^{\text{weighted}}$)
Grade-level tree density is calculated as the total number of trees detected across all polygons in grade $g$, divided by the total land area of those polygons:

$$D_g^{\text{weighted}} = \frac{\sum_{j \in g} N_j}{\sum_{j \in g} A_j} = \sum_{j \in g} \left(\frac{A_j}{\sum_{k \in g} A_k}\right) D_j = \sum_{j \in g} w_j D_j$$

where $w_j = \frac{A_j}{\sum_{k \in g} A_k}$ is the spatial area share of polygon $j$ within grade $g$.

### 6.2 Aggregate Mean Detected Crown Area ($\bar{C}_g^{\text{aggregate}}$)
Grade-level mean crown area is calculated as the sum of all individual crown areas across all trees in grade $g$, divided by the total number of trees in grade $g$:

$$\bar{C}_g^{\text{aggregate}} = \frac{\sum_{j \in g} \sum_{i=1}^{N_j} C_{ij}}{\sum_{j \in g} N_j} = \sum_{j \in g} \left(\frac{N_j}{\sum_{k \in g} N_k}\right) \bar{C}_j$$

This represents a tree-count-weighted mean of polygon crown sizes, ensuring that every detected tree exerts equal weight on the aggregate.

### 6.3 Area-Weighted Grade Crown-Area Fraction ($F_g^{\text{weighted}}$)
Grade-level crown-area fraction is calculated as total detected crown area in grade $g$ divided by total ground area in grade $g$:

$$F_g^{\text{weighted}} = \frac{\sum_{j \in g} \sum_{i=1}^{N_j} C_{ij}}{\sum_{j \in g} \text{Area}_{\text{m}^2, j}} \times 100\% = \sum_{j \in g} w_j F_j$$

### 6.4 Comparison of Weighted Aggregates vs. Unweighted Means
The table below illustrates the substantial divergence between rigorous area-weighted aggregates and naïve unweighted averages:

```
+-------+---------------+-----------------------------------+-----------------------------------+
| Grade | Sample Size   | Tree Density (trees/km²)          | Crown Area Fraction (%)           |
|       |               | Area-Weighted    Naïve Unweighted | Area-Weighted    Naïve Unweighted |
+-------+---------------+-----------------------------------+-----------------------------------+
| A     | N = 3         | 1,477.5          1,065.4          | 0.198%           0.154%           |
| B     | N = 7         |   849.0            791.5          | 0.140%           0.131%           |
| C     | N = 1         |   629.2            629.2          | 0.095%           0.095%           |
| D     | N = 7         |   700.1            769.7          | 0.113%           0.129%           |
+-------+---------------+-----------------------------------+-----------------------------------+
```

In Grade A, unweighted averaging understates tree density by $27.9\%$ because the large, dense Polygon 9093 is weighted equally with two much smaller polygons. In Grade D, unweighted averaging overstates tree density by $9.9\%$ because tiny pocket polygons with local residential tree clusters are given equal weight to large industrial/rail corridors. Naïve unweighted averages are therefore rejected in all reported scientific figures.

---

## 7. Influence and Sensitivity Analysis Methodology

### 7.1 Dominance of Polygon 9093 in Grade A
An essential structural finding of our quality-control audit is the extreme concentration of Grade A land and trees within a single spatial unit:
- **Polygon 9093** (encompassing affluent West Austin, Old West Austin, Pemberton Heights, and Enfield) spans **$12.88\text{ km}^2$**, accounting for **$81.46\%$ of all Grade A land area** ($15.81\text{ km}^2$) and **$90.05\%$ of all Grade A detected trees** (21,034 of 23,357).
- In contrast, the remaining two Grade A polygons (9090 and 9083) are small enclaves ($1.76\text{ km}^2$ and $1.17\text{ km}^2$, respectively).

### 7.2 Exclusion Scenario Formulation
To test whether the apparent "Grade A canopy advantage" is a consistent property of Grade A historical designations or an artifact of Polygon 9093, we compute an explicit exclusion scenario:

$$D_{A \setminus \{9093\}}^{\text{weighted}} = \frac{N_{9090} + N_{9083}}{A_{9090} + A_{9083}} = \frac{1,473 + 850}{1.760781 + 1.170288} = \frac{2,323}{2.931070} = 792.54\text{ trees/km}^2$$

$$F_{A \setminus \{9093\}}^{\text{weighted}} = \frac{\text{Crown}_{9090} + \text{Crown}_{9083}}{\text{Area}_{9090} + \text{Area}_{9083}} \times 100\% = \frac{2,327.47 + 1,345.06}{2,931,069.76} \times 100\% = 0.1253\%$$

### 7.3 Leave-One-Polygon-Out (LOPO) Jackknife Sensitivity Analysis
To systematically evaluate the influence of every historical polygon without making parametric distributional assumptions, we implemented a complete Leave-One-Polygon-Out (LOPO) sensitivity analysis across all 18 polygons.

For each polygon $j$ belonging to grade $g$:
1. The grade aggregate is recalculated excluding polygon $j$:
   $$D_{g \setminus \{j\}} = \frac{\left(\sum_{k \in g} N_k\right) - N_j}{\left(\sum_{k \in g} A_k\right) - A_j}$$
   $$F_{g \setminus \{j\}} = \frac{\left(\sum_{k \in g} \text{Crown}_k\right) - \text{Crown}_j}{\left(\sum_{k \in g} \text{Area}_k\right) - \text{Area}_j} \times 100\%$$
2. The absolute parameter shift is calculated:
   $$\Delta D_{(j)} = D_{g \setminus \{j\}} - D_g^{\text{weighted}}$$
   $$\Delta F_{(j)} = F_{g \setminus \{j\}} - F_g^{\text{weighted}}$$
3. The percentage parameter shift is calculated:
   $$\% \Delta D_{(j)} = \left(\frac{D_{g \setminus \{j\}} - D_g^{\text{weighted}}}{D_g^{\text{weighted}}}\right) \times 100\%$$
   $$\% \Delta F_{(j)} = \left(\frac{F_{g \setminus \{j\}} - F_g^{\text{weighted}}}{F_g^{\text{weighted}}}\right) \times 100\%$$

**Boundary Condition for Grade C:** Because Grade C contains exactly one polygon ($N_C = 1$, Polygon 9084), removing it leaves $0$ trees and $0\text{ km}^2$. The LOPO calculation for Grade C is mathematically undefined ($\frac{0}{0}$) and explicitly designated as such in the sensitivity dataset.

---

## 8. Methodological and Data Limitations

Rigorous peer review requires candid documentation of the inherent spatial, historical, and structural limitations of the Austin HOLC dataset:

### 8.1 Limited Historical Geographic Footprint ($5.30\%$)
The 18 HOLC polygons encompass only $45.68\text{ km}^2$, representing just $5.30\%$ of modern Austin's municipal landmass. Over $94\%$ of the contemporary city (including vast suburban residential tracts, master-planned developments, East Austin peripheral communities, and tech corridors) was never surveyed by the HOLC in 1937. Findings from the HOLC footprint cannot be generalized to the broader metropolitan area.

### 8.2 Extreme Sample Size Imbalance and Degrees of Freedom
The sample sizes across historical grades are severely constrained and highly unbalanced:
- Grade A: $n = 3$
- Grade B: $n = 7$
- Grade C: $n = 1$
- Grade D: $n = 7$
- **Total sample size: $N = 18$ polygons**.

With $N_C = 1$, within-grade sample variance is strictly zero and population variance cannot be estimated. Standard inferential procedures (e.g., ordinary least squares regression, analysis of variance [ANOVA], Kruskal-Wallis non-parametric tests) violate core statistical assumptions regarding sample size, group variance estimation, and normality.

### 8.3 Spatial Autocorrelation and Geographic Clustering
The 18 polygons exhibit intense spatial clustering that violates the assumption of independent and identically distributed (i.i.d.) observations:
- All Grade A polygons are clustered in the western hills along the Colorado River.
- Grade D polygons are clustered in the eastern flatlands, along the Missouri-Kansas-Texas (MKT) and Southern Pacific railroad corridors, and in south-central lowlands.
- This geographic sorting means that historical grade is perfectly confounded with pre-existing physiography (elevation, slope, soils, hydrology, and microclimate).

### 8.4 Temporal Discordance (83-Year Decoupling)
The analysis compares 1937 appraisal boundaries with ~2020 satellite-detected tree canopies, an 83-year temporal separation. Over these eight decades, the study area experienced:
- Widespread urban renewal programs (1950s–1970s) that bulldozed extensive housing and tree cover in East Austin (Grade D).
- The construction of Interstate Highway 35 (completed 1962), which physically severed East Austin from the downtown core.
- Private residential landscape investments, tree maturation, institutional plantings, and modern gentrification.
Contemporary canopy differences reflect cumulative 20th-century urban evolution rather than the isolated, direct impact of 1937 mortgage security maps.

### 8.5 Absence of Raw Multispectral Rasters Locally
The original continuous multispectral satellite scenes and raw segmentation prediction rasters were not available in the local computing environment. The analysis relies entirely on the audited vector point GeoJSON. Consequently, continuous canopy cover texture, sub-pixel foliage density, vertical structure, and tree species composition cannot be directly evaluated.

---

## 9. Scientific Interpretation: Descriptive Association vs. Causal Attribution

### 9.1 Non-Causal Nature of Descriptive Comparisons
**Under no circumstances should the observed tree canopy differences be interpreted as the direct causal effect of HOLC redlining.**

In urban history and spatial economics, attributing modern canopy disparities solely to 1937 HOLC maps commits the fallacy of *post hoc ergo propter hoc*. In Austin, the spatial patterns of racial segregation and environmental inequality were formally codified **nine years prior to the HOLC survey**:
- **The 1928 Austin City Plan (Koch & Fowler):** Formally established a segregated "Negro District" east of East Avenue (modern I-35) by deliberately concentrating all public schools, parks, and health clinics for Black residents in East Austin while denying them city services in other neighborhoods.
- **Topographic and Environmental Sorting:** High-income white residents had already settled in the elevated, well-drained, forested limestone hills of West Austin decades before 1937, whereas industrial facilities, rail yards, and low-income minority housing were sited in the flat, alluvial eastern plains.
- **HOLC as Documentation Rather than Innovation:** As recent historical scholarship demonstrates (e.g., Fishback et al., 2022; Hillier, 2003), HOLC appraisers did not invent these disparities; they documented and codified pre-existing private mortgage practices, racial covenants, and municipal segregation policies.

### 9.2 Recommendation for Manuscript Integration
Given the extreme sample imbalance ($n=1$ for Grade C), the single-polygon dominance of Polygon 9093 in Grade A, and the small geographic coverage ($5.30\%$), we strongly recommend:
1. **Refraining from inferential regression modeling or hypothesis testing** ($p$-values, $F$-tests) on the 18 HOLC polygons.
2. Presenting the Austin HOLC findings strictly as an **exploratory, descriptive case study** in the Supplementary Information or Discussion, explicitly contextualized by the 1928 Master Plan and environmental topography.
3. Highlighting that while Grade A exhibits higher tree density overall, this relationship is **structurally fragile** and entirely contingent upon the inclusion of a single large, affluent polygon (9093). When Polygon 9093 is excluded, Grade A tree density ($792.5\text{ trees/km}^2$) and canopy fraction ($0.125\%$) fall below those of Grade B ($849.0\text{ trees/km}^2$ and $0.140\%$), demonstrating that a monotonic "canopy gradient" across historical grades does not robustly exist in Austin.

---

## References

1. Fishback, P. V., LaVoice, J., Shertzer, A., & Walsh, R. (2022). *Race, risk, and the emergence of Federal redlining*. National Bureau of Economic Research, Working Paper No. 29424.
2. Hillier, A. E. (2003). Spatial analysis of historical redlining: A methodological exploration. *Journal of Housing Research*, 14(1), 137-167.
3. Koch & Fowler. (1928). *A City Plan for Austin, Texas*. City of Austin Planning Commission.
4. Nelson, R. K., Winling, L., Marciano, R., Connolly, N., et al. (2023). *Mapping Inequality: Redlining in New Deal America*. American Panorama, Digital Scholarship Lab, University of Richmond. https://dsl.richmond.edu/panorama/redlining/
5. U.S. Census Bureau. (2020). *TIGER/Line Shapefiles: Census Block Groups and Land Areas, Texas*. U.S. Department of Commerce.
