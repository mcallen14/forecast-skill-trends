# Activate virtual environment
& ..\venv-forecast1\Scripts\Activate.ps1

# Run scripts
python find_nws_usgs_site_pairs.py
python download_usgs_streamflow.py
python download_AR_recon_records.py
python download_cnrfc_forecasts.py
python compile_cnrfc_forecasts.py
python clean_cnrfc_usgs.py
python download_usgs_basinchar.py
python calculate_metric_daily.py
python calculate_metric_aggregated.py
python reliability_sensitivity_analysis.py
python bayesian_analysis_basinchar.py
python plot_results_main.py
