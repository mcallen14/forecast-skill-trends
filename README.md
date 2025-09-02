# Characterizing Improvements in Ensemble Forecast Skill Over the Last Decade: A Retrospective Analysis of the Hydrologic Ensemble Forecast Service (HEFS)

This repository contains code and data processing workflows for analyzing the performance of the **Hydrologic Ensemble Forecast Service (HEFS)** at the **California–Nevada River Forecast Center (CNRFC)**. The analysis evaluates whether operational streamflow forecasts have improved since HEFS implementation in 2014, and what factors influence forecast performance across sites, lead times, and hydrologic regimes. An associated paper has been submitted for peer review, and will be added to this README when available.  

---

## Repository Structure

- `code/` – Scripts for retrieving and processing forecast, observed, and auxiliary datasets; calculating skill metrics and running regression models.
- `analysis/data/` – Where datasets are stored after downloading and cleaning; calculated metrics are also stored within this folder. This folder and all its contents are available from a Zenodo dataset repository (Allen, 2025; DOI: 10.5281/zenodo.17024920).
- `analysis/log/` – Where log of print and error statements for each script are saved.
- `analysis/bayesian_traces/` – Where prior and posterior traces from Bayesian models are stored.
- `analysis/plots/` – Model outputs, tables, and summary statistics.

## Key Scripts

Located within code folder:
- `config.py` – Configuration file for paths, thresholds, and runtime options.

*Data download & cleaning:*
- `find_nws_usgs_site_pairs.py`
- `download_usgs_streamflow.py`
- `download_AR_recon_records.py`
- `download_cnrfc_forecasts.py`
- `compile_cnrfc_forecasts.py`
- `clean_cnrfc_usgs.py`
- `download_usgs_basinchar.py`

*Metric calculation:*
- `calculate_metric_daily.py`
- `calculate_metric_aggregated.py`

*Sensitivity analysis for Reliability metric:*
- `reliability_sensitivity_analysis.py`

*Main Bayesian analysis:*
- `bayesian_analysis_basinchar.py`

*Visualization of results:*
- `plot_results_main.py`


---

## Data

We use four primary datasets:

1. **CNRFC Forecasts (2025)**  
   - Archived operational ensemble forecasts at ~300 sites.  
   - Only 97 sites retained after filtering for record length, missing data, and quality control.  
   - Forecast horizon: up to 14 days ahead.  
   - Daily aggregation (08:00–08:00 GMT) aligned with USGS observed flow.  

2. **USGS Streamflow Observations (2025)**  
   - Daily mean discharge at CNRFC forecast sites.  
   - Retrieved from the [USGS Water Data for the Nation](https://waterdata.usgs.gov/nwis).  

3. **AR Recon Assimilation Records (CW3E, 2025)**  
   - Daily counts of dropsondes assimilated into GEFS (2016–2025).  
   - Summed over the three days preceding each forecast issue date.  

4. **Watershed Characteristics (Falcone, 2011)**  
   - Subset of Gages-II attributes: mean annual precipitation, mean elevation, relief ratio, log drainage area, number of upstream dams, and permeability.

  
---

## Methodology
The following summarize the default methodological workflow used in the analysis; most of these (e.g., lead times and skill metrics of interest) can be modified in the config.py file.

### Research Questions
1. **What aspects of forecast skill have improved?**  
   - Forecast lead times: 1, 3, 5, 7, 10, and 14 days.  
   - Flow magnitudes:  
     - High flows (> 90th percentile)  
     - Moderate-to-high flows (> 50th percentile)  
   - Skill metrics: deterministic and probabilistic (RMSE, eCRPS, Reliability).  

2. **What factors explain variations in forecast performance?**  
   - **Wetness** (average flow in a WY, by flow category)  
   - **AR Recon** (dropsonde assimilation counts)  
   - **GEFSv12** (post-2021 model upgrade indicator)  
   - **Time** (years since HEFS implementation)  

### Forecast Skill Metrics
- **RMSE** – deterministic accuracy of the ensemble mean.  
- **eCRPS** – probabilistic skill, integrating across ensemble spread.  
- **Reliability** – compares the empirical non-exceedance probability (NEP) of the observed flow to a perfectly reliable forecast (i.e., one in which observations appear as random draws from the ensemble forecast) (see Evin et al., 2014; McInerney et al., 2017).
 

A Monte Carlo experiment determined ≥10 events per WY are needed for reliable estimates.  

### Hierarchical Bayesian Regression
We develop a hierarchical Bayesian regression model that partially pools information across sites, allowing for robust detection of performance trends with limited data. This framework supports both local and regional inference and helps isolate the drivers of forecast performance. 


We implement a hierarchical Bayesian framework to quantify trends and predictors of skill. The standardized performance metric at site i and WY t is modeled as:

Metric_it ~ Normal( μ_it , σ_it )

with:

μ_it = β_0i + β_1i * Wetness_it + β_2i * Time + β_3i * ARRecon_it + β_4i * GEFSv12_t

Key additional elements, explained within the scripts and the associated paper (in submission), include:
- Site-level coefficients (beta_ki) are drawn from regional distributions.  
- Variance is scaled by event count per WY.  
- Extended version includes **basin characteristics** as meta-regressors.  
- Fitting via **PyMC** (4 chains, 10k tune, 10k draws).  
- Convergence checked with trace plots and R-hat.  

---

## Getting Started

1. **Clone the repo:**
   ```bash
   git clone https://github.com/mcallen14/forecast-skill-trends.git
   cd forecast-skill-trends
   ```

2. **Set up the virtual environment:**
  ```bash
  bash setup_venv-forecast1_withpymc.sh
  ```

3. **Get data:**

Data can be accessed in three different ways: 
A) Download from a Zenodo dataset repository (Allen, 2025; DOI: 10.5281/zenodo.17024920), unzip the 'data' folder from the Zenodo repo in the 'analysis' folder associated with this Github repo. This option is the most straightforward if you do not need to extend the anlaysis to sites beyond the 97 associated with the current study.
B) Use the data download & cleaning scripts in this Github repo (run the python scripts in the order listed in the Key Scripts section above).
C) Access via the URLs in associated references below (though this option is not recommended, as the exact format and structure of the data may vary slightly from the ones above, making subsequent analysis more difficult).


3. **Run analysis and visualize results:**

If you chose option A above (downloaded data via Zenodo), then the daily and aggregated forecast skill metrics have already been calculated, and you can run the remaining two primary scripts in the workflow (`bayesian_analysis_basinchar.py` and `plot_results_main.py`).

If you chose option B above and have run all of the scripts in the *Data download & cleaning* section of Key Scripts, then you'll need to run the metric calculation scripts followed by the Bayesian analysis and results visualization scripts (in order: `calculate_metric_daily.py`, `calculate_metric_aggregated.py`, `bayesian_analysis_basinchar.py`, and `plot_results_main.py`).

To see the results of the Monte Carlo simulation described in the supplementary information, run `reliability_sensitivity_analysis.py`.

To run all of the scripts, you can also run the following shell script:
  ```bash
  bash run_all_scripts.sh
  ```
---

## References

Allen, M. C. (2025). Streamflow Forecasts, Observations, Basin Characteristics, AR Recon Records, and Forecast Skill Metrics for Sites in California, Nevada, and Oregon (Water Years 2014–2025). [Dataset]. https://doi.org/10.5281/ZENODO.17024920

CNRFC. (2025a). California Nevada River Forecast Center (CNRFC) Short Range Hourly Ensemble Data Archive. [Dataset] Retrieved from https://www.cnrfc.noaa.gov/ensembleHourlyProductCSV.php on June 20, 2025.

CW3E. (2025). Center for Western Weather and Water Extremes (CW3E) Atmospheric River Reconnaissance daily AR data collection and integration. [Data set]. Retrieved from https://cw3e.ucsd.edu/arrecon_overview/ on July 16, 2025.

Evin, G., Thyer, M., Kavetski, D., McInerney, D., & Kuczera, G. (2014). Comparison of joint versus postprocessor approaches for hydrological uncertainty estimation accounting for error autocorrelation and heteroscedasticity. Water Resources Research, 50(3), 2350–2375. https://doi.org/10.1002/2013WR014185

Falcone, J. (2011). Geospatial Attributes of Gages for Evaluating Streamflow: U.S.  Geological Survey data release. [Data set]. https://doi.org/10.5066/P96CPHOT

McInerney, D., Thyer, M., Kavetski, D., Lerat, J., & Kuczera, G. (2017). Improving probabilistic prediction of daily streamflow by identifying Pareto optimal approaches for modeling heteroscedastic residual errors. Water Resources Research, 53(3), 2199–2239. https://doi.org/10.1002/2016WR019168

USGS. (2025). Daily streamflow data from the U.S. Geological Survey National Water Information System. [Dataset]. Accessed via dataretrieval package July 17, 2025.

Wilks, D. S. (2019). Statistical Methods in the Atmospheric Sciences, Fourth Edition. Statistical Methods in the Atmospheric Sciences, Fourth Edition, 1–818. https://doi.org/10.1016/C2017-0-03921-6
