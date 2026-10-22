# Data Sources and Citations
**Analytical Scope:** Austin, TX; Bloomington, IN; Cupertino, CA; Surrey, BC (1,283 Census Units)

---

## 1. Remote Sensing & Optical Earth Observation Imagery

### 1.1 Maxar Very-High-Resolution (VHR) Commercial Satellite Constellation
- **Author / Provider:** Maxar Technologies (formerly DigitalGlobe, Inc.) / National Geospatial-Intelligence Agency (NGA) / NASA Commercial SmallSat Data Acquisition (CSDA) Program.
- **Title / Product:** High-Resolution Commercial Satellite Imagery (QuickBird-2, GeoEye-1, WorldView-2, and WorldView-3).
- **Temporal Coverage:** 2006–2024 (peak vegetative season, May–November).
- **Spatial Resolution:** 0.31–0.50 m panchromatic; 1.24–2.00 m multispectral (orthorectified and pan-sharpened to 0.50 m ground sampling distance).
- **Access / Licensing:** Accessed via the NextView License agreement administered through the National Geospatial-Intelligence Agency (NGA) and NASA Commercial SmallSat Data Acquisition (CSDA) program.
- **Formal Citation (Nature / APA):**  
  > Maxar Technologies. (2024). *Maxar Very High Resolution Optical Satellite Constellation: QuickBird-2, GeoEye-1, WorldView-2, and WorldView-3 Imagery* [Data set]. National Geospatial-Intelligence Agency NextView License / NASA Commercial SmallSat Data Acquisition (CSDA) Program. https://www.maxar.com/
- **Methodological Reference:**  
  > Tucker, C. J., Brandt, M., Hiernaux, P., Kariryaa, A., Rasmussen, K., Small, J., Romero-Sanchez, C., Igel, C., & Fensholt, R. (2023). Sub-continental-scale carbon stocks of individual trees in African drylands. *Nature*, 615(7950), 80–86. https://doi.org/10.1038/s41586-022-05653-6
  > 
  > Brandt, M., Tucker, C. J., Kariryaa, A., Rasmussen, K., Hiernaux, P., Igel, C., Mugnier, M. L., & Fensholt, R. (2020). An unexpectedly large count of trees in the West African Sahara and Sahel. *Nature*, 587(7832), 78–82. https://doi.org/10.1038/s41586-020-2824-5

---

## 2. Demographic & Socioeconomic Census Data

### 2.1 United States: American Community Survey (ACS) 5-Year Estimates (2019–2023)
- **Author / Agency:** U.S. Census Bureau, U.S. Department of Commerce.
- **Survey:** American Community Survey (ACS) 5-Year Data (2019–2023).
- **Geographic Level:** Census Block Groups (Austin, TX; Bloomington, IN; Cupertino, CA).
- **Tables Harvested:**
  - **Table B19013:** *Median Household Income in the Past 12 Months (in 2023 Inflation-Adjusted Dollars)* (`ACSDT5Y2023.B19013`). Variable: `B19013_001E` (Estimate) and `B19013_001M` (Margin of Error).
  - **Table B25001:** *Housing Units* (`ACSDT5Y2023.B25001`). Used to compute neighborhood gross housing density (dwellings km⁻²).
  - **Table B25035:** *Median Year Structure Built* (`ACSDT5Y2023.B25035`). Used for urban development epoch controls.
  - **Table B03002:** *Hispanic or Latino Origin by Race* (`ACSDT5Y2023.B03002`). Demographic composition covariates.
- **Formal Citation (Nature / APA):**  
  > U.S. Census Bureau. (2024). *2019–2023 American Community Survey 5-Year Estimates, Tables B19013 (Median Household Income), B25001 (Housing Units), B25035 (Median Year Built), and B03002 (Race/Ethnicity)* [Data set]. U.S. Department of Commerce. https://data.census.gov/

### 2.2 Canada: Statistics Canada 2021 Census of Population
- **Author / Agency:** Statistics Canada, Government of Canada.
- **Census:** 2021 Census of Population.
- **Geographic Level:** Dissemination Areas (DAs) for the City of Surrey, British Columbia (Census Division 5915, Census Subdivision 5915004).
- **Catalogue Number:** 98-401-X2021006 / 98-316-X2021001. Released December 15, 2022.
- **Variables Harvested:**
  - *Median total income of household in 2020 ($)* (Characteristic ID 113).
  - *Total private dwellings* (Characteristic ID 2) for housing density (dwellings km⁻²).
- **Currency Normalization:** Bank of Canada 2023 Annual Average Daily Exchange Rate (1.37 CAD = 1.00 USD).
- **Formal Citation (Official Statistics Canada Citation):**  
  > Statistics Canada. (2022). *Census Profile, 2021 Census of Population: Canada, Provinces, Territories, Census Divisions, Census Subdivisions and Dissemination Areas* (Statistics Canada Catalogue no. 98-401-X2021006 / 98-316-X2021001) [Data set]. Government of Canada, Ottawa. Released December 15, 2022. https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/index.cfm?Lang=E

---

## 3. Cartographic & Administrative Geometries

### 3.1 United States: 2023 TIGER/Line Shapefiles
- **Author / Agency:** U.S. Census Bureau, Geography Division.
- **Release:** 2023 TIGER/Line Shapefiles.
- **Products & Feature Datasets:**
  - *Census Block Groups (Current):*
    - Texas (`tl_2023_48_bg.zip`)
    - Indiana (`tl_2023_18_bg.zip`)
    - California (`tl_2023_06_bg.zip`)
  - *Incorporated Places / Legal Municipal Boundaries:*
    - Texas Places (`tl_2023_48_place.zip`, FIPS 05000 - Austin)
    - Indiana Places (`tl_2023_18_place.zip`, FIPS 05860 - Bloomington)
    - California Places (`tl_2023_06_place.zip`, FIPS 17610 - Cupertino)
- **Formal Citation:**  
  > U.S. Census Bureau. (2023). *2023 TIGER/Line Shapefiles: Current State-based Census Block Groups (tl_2023_06_bg, tl_2023_18_bg, tl_2023_48_bg) and Incorporated Places (tl_2023_06_place, tl_2023_18_place, tl_2023_48_place)* [Digital spatial data]. U.S. Department of Commerce. https://www2.census.gov/geo/tiger/TIGER2023/

### 3.2 Canada: 2021 Cartographic Boundary Files
- **Author / Agency:** Statistics Canada, Geography Division.
- **Release:** 2021 Census Cartographic Boundary Files (CBF).
- **Products:**
  - *Dissemination Areas (Cartographic Boundary File):* `lda_000b21a_e.shp` (Catalogue no. 92-169-X).
  - *Census Subdivisions (Cartographic Boundary File):* `lcsd000b21a_e.shp` (Catalogue no. 92-160-X).
- **Formal Citation:**  
  > Statistics Canada. (2021). *2021 Census Cartographic Boundary Files: Dissemination Areas (lda_000b21a_e, Cat. No. 92-169-X) and Census Subdivisions (lcsd000b21a_e, Cat. No. 92-160-X)* [Digital spatial data]. Government of Canada, Ottawa. https://www150.statcan.gc.ca/n1/en/catalogue/92-169-X

---

## 4. Official Municipal Parkland & Open-Space GIS Layers (Reviewer 2 Covariates)

### 4.1 City of Austin, Texas
- **Issuing Agency:** City of Austin Parks and Recreation Department (PARD) / City of Austin Communications and Technology Management.
- **Layer Title:** *City of Austin Parks and Recreation Department Parkland*
- **Identifier:** Socrata Dataset ID `8f2b-a4q5` (Internal GIS Resource: `v8hw-gz65`).
- **Data Attributes Used:** Polygon boundaries, `tpl_landuse` ("Natural Lands"), `park_type` ("Nature Preserve", "Greenbelt").
- **Formal Citation:**  
  > City of Austin Parks and Recreation Department. (2024). *City of Austin Parks and Recreation Department Parkland* [GIS polygon dataset, Dataset ID: 8f2b-a4q5]. City of Austin Open Data Portal. https://data.austintexas.gov/d/8f2b-a4q5

### 4.2 City of Bloomington, Indiana
- **Issuing Agency:** City of Bloomington Parks and Recreation Department / Information Technology Services (ITS).
- **Layer Title:** *Bloomington Parks*
- **Identifier:** ArcGIS REST MapServer `DataPortal_ParksAndRec`, Layer ID `13` (Parks).
- **Data Attributes Used:** Polygon boundaries, `type` ("Nature Preserve", "Community Park", "Neighborhood Park").
- **Formal Citation:**  
  > City of Bloomington Parks and Recreation Department. (2024). *City of Bloomington Parks* [GIS polygon layer 13, DataPortal_ParksAndRec MapServer]. City of Bloomington GeoHub Open Data Portal. https://bloomington.in.gov/geohub

### 4.3 City of Cupertino, California
- **Issuing Agency:** City of Cupertino GIS Division / Public Works Department.
- **Layer Title:** *City of Cupertino Parks*
- **Identifier:** ArcGIS REST MapServer `Public/PropertyInfo_Internal2`, Layer ID `22`.
- **Data Attributes Used:** Polygon boundaries, `EnvironmentalArea` ("Yes" / "1" / "True").
- **Formal Citation:**  
  > City of Cupertino GIS Division. (2024). *City of Cupertino Parks* [GIS polygon layer 22, PropertyInfo_Internal2 MapServer]. City of Cupertino GIS Portal. https://gis.cupertino.org

### 4.4 City of Surrey, British Columbia
- **Issuing Agency:** City of Surrey Corporate GIS / Parks, Recreation & Culture Department.
- **Layer Titles:**
  - *Surrey Parks* (ArcGIS MapServer `OpenData`, Layer ID `256`)
  - *Park Natural Areas* (ArcGIS MapServer `OpenData`, Layer ID `245`)
- **Data Attributes Used:** Municipal park boundaries and dedicated natural-area conservation polygons.
- **Formal Citation:**  
  > City of Surrey Corporate GIS. (2024). *Surrey Parks (Layer 256) and Park Natural Areas (Layer 245)* [GIS polygon datasets, OpenData MapServer]. City of Surrey Open Data. https://data.surrey.ca/dataset/parks

---

## 5. Municipal Ground-Truth & Urban Tree Inventories (Contextual Reference)

### 5.1 Cupertino Public Tree Inventory
- **Issuing Agency:** City of Cupertino Public Works Department.
- **Dataset:** *Cupertino Tree Inventory* [GIS point dataset].
- **URL:** https://gis-cupertino.opendata.arcgis.com

### 5.2 Surrey Park Specimen Tree Inventory
- **Issuing Agency:** City of Surrey Parks Division.
- **Dataset:** *Park Specimen Trees* [GIS point dataset].
- **URL:** https://data.surrey.ca/dataset/park-specimen-trees

---

## 6. Econometric & Spatial Statistical Methodology

- **Heteroskedasticity-Consistent Covariance (HC3):**
  > MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305–325. https://doi.org/10.1016/0304-4076(85)90044-8
  > 
  > Long, J. S., & Ervin, L. H. (2000). Using heteroscedasticity consistent standard errors in the linear regression model. *The American Statistician*, 54(3), 217–224. https://doi.org/10.1080/00031305.2000.10474549

- **Spatial Autoregressive Error Models (SEM) & Spatial Contiguity:**
  > Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic Publishers, Dordrecht. https://doi.org/10.1007/978-94-015-7799-1
  > 
  > Ord, J. K. (1975). Estimation methods for models of spatial interaction. *Journal of the American Statistical Association*, 70(349), 120–126. https://doi.org/10.1080/01621459.1975.10480272

- **Multiple Hypothesis Testing Correction (False Discovery Rate):**
  > Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

- **Spatial Point Pattern Dispersion (Ripley’s K):**
  > Ripley, B. D. (1977). Modelling spatial patterns. *Journal of the Royal Statistical Society: Series B (Methodological)*, 39(2), 172–192. https://doi.org/10.1111/j.2517-6161.1977.tb01615.x

---

## 7. Open-Source Scientific Computational Software

- **PySAL (Python Spatial Analysis Library):**
  > Rey, S. J., & Anselin, L. (2007). PySAL: A Python library of spatial analytical methods. *The Review of Regional Studies*, 37(1), 5–27.
  > 
  > Rey, S. J., Anselin, L., Amaral, P. F., Arribas-Bel, D., Cortes, X., Gaboardi, J. D., Kang, W., Knaap, E., Li, Z., Lumnitz, S., Oshan, T. M., Shao, H., & Wolf, L. J. (2021). The PySAL Ecosystem: Philosophy and Implementation. *Geographical Analysis*, 54(3), 467–487. https://doi.org/10.1111/gean.12319

- **GeoPandas:**
  > Jordahl, K., Bossche, J. V. den, Fleischmann, M., Wasserman, J., McBride, J., Gerard, J., Tratner, J., Perry, M., Badaracco, A. G., Bell, R., Snow, A. D., et al. (2020). *geopandas/geopandas: v0.8.1* (Version v0.8.1). Zenodo. https://doi.org/10.5281/zenodo.3946761

- **Statsmodels:**
  > Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with python. In *Proceedings of the 9th Python in Science Conference* (Vol. 57, pp. 61–66).

- **SciPy:**
  > Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2

- **Scikit-image:**
  > van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., & Yu, T. (2014). scikit-image: image processing in Python. *PeerJ*, 2, e453. https://doi.org/10.7717/peerj.453

---

## 8. Data Availability Statement

> **Data Availability**  
> High-resolution optical satellite imagery (QuickBird-2, GeoEye-1, WorldView-2, and WorldView-3) was acquired through the NASA Commercial SmallSat Data Acquisition (CSDA) Program via the National Geospatial-Intelligence Agency (NGA) NextView License (https://www.maxar.com/). U.S. socioeconomic data and cartographic boundary shapefiles are publicly available from the U.S. Census Bureau via data.census.gov and the TIGER/Line FTP service (2019–2023 ACS 5-Year Estimates, Tables B19013, B25001, B25035, and B03002; https://www2.census.gov/geo/tiger/TIGER2023/). Canadian socioeconomic data and cartographic boundary files are publicly available from Statistics Canada (2021 Census Profile, Catalogue no. 98-401-X2021006; Dissemination Area Boundary Files, Catalogue no. 92-169-X; https://www12.statcan.gc.ca/census-recensement/). Official municipal parkland and natural-area GIS layers are publicly accessible via the respective open-data portals of the City of Austin (https://data.austintexas.gov/d/8f2b-a4q5), City of Bloomington (https://bloomington.in.gov/geohub), City of Cupertino (https://gis.cupertino.org), and City of Surrey (https://data.surrey.ca/dataset/parks). Derived census-unit tree metrics, urban covariates, and model estimation tables generated in this study are available in the project repository and Supplementary Data files.
