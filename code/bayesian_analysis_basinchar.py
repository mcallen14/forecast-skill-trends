"""
Steinschneider Lab Group
Madeline Allen
Code last updated: 7/16/2025
Script purpose: conduct Bayesian hierarcical model analysis

This project utilizes code generated with the assistance of OpenAI's ChatGPT (July, 2025). 
"""

#import necessary packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pymc as pm
import gc
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
# local
import config
from config import MyLogger

# Set up logger
current_script_name = Path(__file__).stem
logger = MyLogger(config.log_path / f'log_{current_script_name}.txt')

# Set random seed
RANDOM_SEED = 14

# Variables from config file:
sd = config.start_date # analysis start date
ed= config.end_date # analysis end date
years = config.years
sd_water_year = config.sd_water_year
ed_water_year = config.ed_water_year
water_years = config.water_years
window_month_start = config.window_month_start # beginning of aggregation window (extended wet season)
window_month_end = config.window_month_end # end of aggregation window (extended wet season)
flow_thresholds = config.flow_thresholds # flow thresholds considered (e.g., 90th percentile flow and above)
thresholds_string = ''.join(map(str, flow_thresholds))
lead_times=config.selected_lead_times # lead times to calculate metrics for
lead_string = ''.join(map(str, lead_times))
AR_recon_window = config.AR_recon_window
aggregation_periods = config.aggregation_periods
selected_metrics = config.selected_metrics
selected_metrics_string = config.selected_metrics_string
event_threshold = config.event_threshold # threshold for min num events included in analysis

# paths
metric_filepath = config.metric_folder
bayesian_traces_path = config.bayesian_traces_path
ID_table_basin_char = pd.read_csv(config.ID_Table_Basin_Char)


# Function for scaling variables:
def scale_by_site_lead(group):
    # Scale the relevant columns with separate StandardScaler instances
    group['Qo_mean_scaled'] = StandardScaler().fit_transform(group[['Qo_mean']])
    group['AR_recon_scaled'] = StandardScaler().fit_transform(group[['AR_recon']])
    
    # Manually scale water_year so it's consistent across groups
    group['water_year_scaled'] = (group['water_year'] - config.sd_water_year)

    # Scale rel by only subtracting mean:
    group[f'rel_scaled'] = StandardScaler(with_mean=True, with_std = False).fit_transform(group[['rel']])
    
    # make new list of selected_metrics without 'rel' if it's in that list:
    selected_metrics_revised = [m for m in selected_metrics if m != 'rel']

    # Loop through selected_metrics except for rel and scale:
    for met in selected_metrics_revised:
        group[f'{met}_scaled'] = StandardScaler().fit_transform(group[[met]])  # Dynamically create column name
    
    return group


#def perform_Bayesian_reg(wet=False, time=False, AR=False, GEFS=False, model_name="", folder_for_traces="", basin_char=False):
def perform_Bayesian_reg(
    wet=False,
    time=False,
    AR=False,
    GEFS=False,
    model_name=None,
    folder_for_traces=None,
    basin_char=False,
    cnrfc_sites=None,
    filt_df=None,
    site=None,
    Qo_mean_scaled=None,
    AR_recon_scaled=None,
    water_year_scaled=None,
    GEFSv12=None,
    num_events_log=None,
    met_scaled=None,
):
    """
    Define and sample from a hierarchical Bayesian regression model.
    
    If basin_char=True, the prior mean for the wetness slope (mu_wet) is modeled as a linear combination
    of site-level basin characteristics (meta-regression structure).
    
    Inputs:
    - wet, time, AR, GEFS: whether to include each respective predictor
    - model_name: name used to save the trace
    - folder_for_traces: directory to save model output
    - basin_char: whether to include one additional level for mu_wet as a function of basin characteristics
    """

    # Set up coordinates
    coords = {
        "site": cnrfc_sites,
        "obs_id": np.arange(len(filt_df)),
    }

    if basin_char:
        coords["feature"] = config.basin_char_names

    with pm.Model(coords=coords) as model:
        # Set mutable data for predictors (obs-level)
        site_idx = pm.Data("site_idx", site, dims="obs_id")
        Qo_idx = pm.Data("Qo_idx", Qo_mean_scaled, dims="obs_id")
        time_idx = pm.Data("time_idx", water_year_scaled, dims="obs_id")
        AR_idx = pm.Data("AR_idx", AR_recon_scaled, dims="obs_id")
        GEFS_idx = pm.Data("GEFS_idx", GEFSv12, dims="obs_id")
        num_events_log_idx = pm.Data("num_events_log_idx", num_events_log, dims="obs_id")

        # Priors for global distributions
        normal_dist_mu = 0.0
        normal_dist_sigma = 3.0
        lognormal_dist_mu = 0.0
        lognormal_dist_sigma = 1.0
        halfcauchy_dist_beta = 1.0


        if basin_char:
            # Basin characteristic column names
            basin_char_columns = config.basin_char_columns 

            # Create DataFrame with only the sites in this model iteration
            X_basin_df = (
                ID_table_basin_char[ID_table_basin_char['CNRFC_id'].isin(cnrfc_sites)]
                .set_index('CNRFC_id')
                .loc[cnrfc_sites, basin_char_columns]
                .astype(float)
            )

            # Scale columns (z-score)
            X_basin_scaled = (X_basin_df - X_basin_df.mean()) / X_basin_df.std(ddof=1) # ddof=1 gives sample std
            # Add intercept column (all 1s)
            X_basin_scaled["intercept"] = 1.0
            X_basin = X_basin_scaled.values

            # Add to model as mutable site-level data
            pm.Data("X_basin", X_basin, dims=("site", "feature"))

            # Prior over meta-regression coefficients
            gamma_wet = pm.Normal("gamma_wet", mu=normal_dist_mu, sigma=normal_dist_sigma, dims="feature")
            gamma_time = pm.Normal("gamma_time", mu=normal_dist_mu, sigma=normal_dist_sigma, dims="feature")


        # Site-specific intercepts (not drawn from a global prior)
        alpha = pm.Normal("alpha", mu=normal_dist_mu, sigma=normal_dist_sigma, dims="site")

        # Define slope for wetness predictor
        if wet:
            if basin_char:
                # mu_wet as linear combination of basin predictors: shape (site,)
                #mu_wet = pm.Deterministic("mu_wet", pm.math.dot(pm.ConstantData("X_basin"), gamma), dims="site")
                mu_wet = pm.Deterministic("mu_wet", pm.math.dot(X_basin, gamma_wet), dims="site")

            else:
                mu_wet = pm.Normal("mu_wet", mu=normal_dist_mu, sigma=normal_dist_sigma)

            sigma_wet = pm.LogNormal("sigma_wet", mu=lognormal_dist_mu, sigma=lognormal_dist_sigma)

            # Site-specific slope on Qo
            beta_wet = pm.Normal("beta_wet", mu=mu_wet, sigma=sigma_wet, dims="site")

        # Define slope for time trend
        if time:
            if basin_char:
                mu_time = pm.Deterministic("mu_time", pm.math.dot(X_basin, gamma_time), dims="site")

            else:
                mu_time = pm.Normal("mu_time", mu=normal_dist_mu, sigma=normal_dist_sigma)
            
            
            sigma_time = pm.LogNormal("sigma_time", mu=lognormal_dist_mu, sigma=lognormal_dist_sigma)
            beta_time = pm.Normal("beta_time", mu=mu_time, sigma=sigma_time, dims="site")

        # Define slope for AR predictor
        if AR:
            mu_AR = pm.Normal("mu_AR", mu=normal_dist_mu, sigma=normal_dist_sigma)
            sigma_AR = pm.LogNormal("sigma_AR", mu=lognormal_dist_mu, sigma=lognormal_dist_sigma)
            beta_AR = pm.Normal("beta_AR", mu=mu_AR, sigma=sigma_AR, dims="site")

        # Define slope for GEFS predictor
        if GEFS:
            mu_GEFS = pm.Normal("mu_GEFS", mu=normal_dist_mu, sigma=normal_dist_sigma)
            sigma_GEFS = pm.LogNormal("sigma_GEFS", mu=lognormal_dist_mu, sigma=lognormal_dist_sigma)
            beta_GEFS = pm.Normal("beta_GEFS", mu=mu_GEFS, sigma=sigma_GEFS, dims="site")

        # Expected value of the outcome
        y_hat = alpha[site_idx]

        if wet:
            y_hat += beta_wet[site_idx] * Qo_idx

        if time:
            y_hat += beta_time[site_idx] * time_idx

        if AR:
            y_hat += beta_AR[site_idx] * AR_idx

        if GEFS:
            y_hat += beta_GEFS[site_idx] * GEFS_idx

        logger.log(y_hat)

        # Model error variance dependent on number of events
        ln_sigma_y_0 = pm.Normal("ln_sigma_y_0", mu=normal_dist_mu, sigma=normal_dist_sigma)
        ln_sigma_y_num_events = pm.Normal("ln_sigma_y_num_events", mu=normal_dist_mu, sigma=normal_dist_sigma)
        sigma_y = pm.Deterministic("sigma_y", pm.math.exp(ln_sigma_y_0 + ln_sigma_y_num_events * num_events_log_idx))

        # Likelihood
        y_like = pm.Normal("y_like", mu=y_hat, sigma=sigma_y, observed=met_scaled, dims="obs_id")

        # Sample posterior
        trace = pm.sample(
            draws=config.posterior_samples,
            tune=config.tuning_samples,
            chains= config.number_chains,
            #cores=1,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True}
        )

        # Sample prior predictive
        prior = pm.sample_prior_predictive(samples=config.prior_samples)

        # Save traces
        trace_file_name = folder_for_traces / f"{model_name}_trace.nc"
        prior_file_name = folder_for_traces /  f"{model_name}_prior.nc"
        trace.to_netcdf(trace_file_name)
        prior.to_netcdf(prior_file_name)
         # CLEAN UP MEMORY
        del trace
        del prior
        del model
        pm.modelcontext(None)  # clear PyMC context
        gc.collect()


if __name__ == "__main__":

    # Define model configurations
    model_configs = config.model_configs

    # Tracking DataFrame for failed model runs
    model_failures = pd.DataFrame(columns=["agg_type", "flow", "lead", "metric", "model_name", "basin_char", "error_message"])


    for agg_type in aggregation_periods: #'window' and/or 'water_year'
        for flow in flow_thresholds:
            logger.log(f"\nProcessing agg period {agg_type} for flow threshold {flow}...")
            #make folder to hold the results from this loop
            folder_name = bayesian_traces_path / f'traces_{agg_type}_Thresh{flow}'
            folder_name.mkdir(parents=True, exist_ok=True)

            # Read in aggregated metric file
            filename = metric_filepath / f"metrics_{agg_type}_Thresh{flow}.csv"
            df_metric = pd.read_csv(filename)

            # filter df based on event threshold determined in monte carlo analysis
            df_metric = df_metric[df_metric['num_events']>=event_threshold]

            # # Temporarily add line there that filters df_metric 'water_year' column only includes 2014 through 2024 (not 2025)
            # df_metric = df_metric[(df_metric['water_year'] >= 2014) & (df_metric['water_year'] <= 2024)]

            # Specify sites present
            selected_sites = df_metric['site'].unique().tolist()

            # Group by 'site' and 'lead', then apply scaling
            #df_metric_scaled = df_metric.groupby(['site', 'lead']).apply(scale_by_site_lead)
            df_metric_scaled = df_metric.groupby(['site', 'lead'], group_keys=False).apply(scale_by_site_lead)

            logger.log(df_metric_scaled.head())

            # Loop through sites and lead times, perform bayesian analysis, save traces:
            for met in selected_metrics:
                for lead in lead_times:
                    logger.log(f"\nProcessing lead {lead} for metric {met}...")

                    # Filter data for the current lead time
                    filt_df = df_metric_scaled[df_metric_scaled['lead'] == lead].copy()

                    # Extract the scaled predictors
                    Qo_mean_scaled = filt_df.Qo_mean_scaled.values
                    AR_recon_scaled = filt_df.AR_recon_scaled.values
                    water_year_scaled = filt_df.water_year_scaled.values
                    GEFSv12 = filt_df.GEFSv12.values
                    num_events = filt_df.num_events.values
                    num_events_log = np.log(num_events)
                    met_scaled = filt_df[f'{met}_scaled'].values  # Dynamically get the scaled metric
                    
                    # Set coords:
                    site, cnrfc_sites = filt_df.site.factorize()

                    # Loop through each model config and run Bayesian regression
                    for model_name_root, args in model_configs.items():
                        model_name = f"{met}_{model_name_root}_ld{lead}"  # Dynamically name the model result
                        logger.log(f"Running {model_name}...")

                        # Store results using dynamically generated key
                        #model_results[model_name] = perform_Bayesian_reg(**args, model_name=model_name)
                        # perform bayesian reg - make basin_char False here
                        try:
                            perform_Bayesian_reg(
                                **args, 
                                model_name=model_name, 
                                folder_for_traces = folder_name, 
                                basin_char=False, 
                                cnrfc_sites=cnrfc_sites,
                                filt_df=filt_df,
                                site=site,
                                Qo_mean_scaled=Qo_mean_scaled,
                                AR_recon_scaled=AR_recon_scaled,
                                water_year_scaled=water_year_scaled,
                                GEFSv12=GEFSv12,
                                num_events_log=num_events_log,
                                met_scaled=met_scaled
                                )
                        except Exception as e:
                            logger.log(f"ERROR running model {model_name}: {e}")
                            # keep track of models that fail
                            model_failures = pd.concat([
                                model_failures,
                                pd.DataFrame([{
                                    "agg_type": agg_type,
                                    "flow": flow,
                                    "lead": lead,
                                    "metric": met,
                                    "model_name": model_name,
                                    "basin_char": False,
                                    "error_message": str(e)
                                }])
                            ], ignore_index=True)

                            continue
                        if config.basin_characteristics == True: #but just do this for the model that has wetness and time.
                            if model_name_root == "wet_time":
                                folder_name_basinchar = bayesian_traces_path / f'traces_{agg_type}_Thresh{flow}_basinchar'
                                folder_name_basinchar.mkdir(parents=True, exist_ok=True)
                                # perform bayesian reg - make basin_char True here
                                try:
                                    perform_Bayesian_reg(
                                        **args, 
                                        model_name=model_name, 
                                        folder_for_traces = folder_name_basinchar, 
                                        basin_char=True, 
                                        cnrfc_sites=cnrfc_sites,
                                        filt_df=filt_df,
                                        site=site,
                                        Qo_mean_scaled=Qo_mean_scaled,
                                        AR_recon_scaled=AR_recon_scaled,
                                        water_year_scaled=water_year_scaled,
                                        GEFSv12=GEFSv12,
                                        num_events_log=num_events_log,
                                        met_scaled=met_scaled
                                        )

                                except Exception as e:
                                    logger.log(f"ERROR running basin char model {model_name}: {e}")
                                    # keep track of models that fail
                                    model_failures = pd.concat([
                                        model_failures,
                                        pd.DataFrame([{
                                            "agg_type": agg_type,
                                            "flow": flow,
                                            "lead": lead,
                                            "metric": met,
                                            "model_name": model_name,
                                            "basin_char": True,
                                            "error_message": str(e)
                                        }])
                                    ], ignore_index=True)
                                    continue

    # Save failures to CSV
    failures_path = bayesian_traces_path / f"model_failures_summary.csv"
    model_failures.to_csv(failures_path, index=False)


    logger.end()