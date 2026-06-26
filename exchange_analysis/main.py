"""
Project: European Electricity Exchange Analysis
Author: Tiernan Buckley
Year: 2026
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Source: https://github.com/INATECH-CIG/exchange_analysis

Description:
Orchestrates the execution of the entire data pipeline, acting as the main control panel 
to trigger downloading, processing, analyzing, and aggregating the grid data.
"""

from datetime import datetime, timedelta, timezone
from entsoe import EntsoePandasClient
from exchange_analysis.config import PipelineConfig
from exchange_analysis.utils import setup_logging, _sync_time_index
import logging
from typing import Optional
from exchange_analysis.BulkDownload import EntsoeFileClientAdapter
from prefect import flow
import pickle

# --- MODULE IMPORTS ---
from exchange_analysis.download_data import (
    download_generation_demand,
    process_generation_demand,
    download_flows,
    process_flows,
    balance_flows_symmetry,
    fetch_simple_metrics
)
from exchange_analysis.data_analysis import (
    perform_decomposition_analysis, 
    perform_aggregated_flow_tracing,
    perform_direct_flow_tracing,
    perform_pooling_analysis,
    perform_post_processing_aggregation
)

def main(start_time: Optional[datetime] = None,
         end_time: Optional[datetime] = None,
         year: Optional[int] = None,
         schema_name: Optional[str] = 'historic-entsoe',
         debug_mode: Optional[bool] = False,
         load_pickle = False,
         save_pickle = False):
    # ==========================================
    # CONTROL PANEL
    # ==========================================

    # 1. Execution Flags (True = Run this step)
    my_run_flags = {
        "download": True,
        "process": True,
        "analysis": True,
        "post_processing": True,
    }

    analysis_subset = {
        "zone_to_gen_type_analysis": True,
        "ac_flow_tracing_analysis": True,
        "dc_flow_tracing_analysis": True,
        "pooling_analysis": True,
    }

    # 2. Define Period
    if year is None:
        year = datetime.now(timezone.utc).year

    if start_time is None or end_time is None:
        start_time = datetime(year, 1, 1, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(year, 12, 31, 23, 59, tzinfo=timezone.utc)
        logger_info_msg = f"Using default full year: {year}"
    else:
        logger_info_msg = f"Using given time range: {start_time} bis {end_time}"

    # Sicherstellen, dass Zeitstempel immer timezone-aware sind
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    period = (
        start_time.strftime("%Y-%m-%d %H:%M"),
        end_time.strftime("%Y-%m-%d %H:%M"))

    # 5. Initialize Config
    config = PipelineConfig(
        date_range=period,
        run_flags=my_run_flags,
        analysis_flags=analysis_subset,
        debug_mode=debug_mode,
        raw_db_schema_name= f"{schema_name}_raw",
        processed_db_schema_name= schema_name
    )

    if load_pickle:
        with open("ioHandler.pkl", "rb") as f:
            io = pickle.load(f)

        config = PipelineConfig(
            date_range=period,
            run_flags=my_run_flags,
            analysis_flags=analysis_subset,
            debug_mode=debug_mode,
            raw_db_schema_name=f"{schema_name}_raw",
            processed_db_schema_name=schema_name,
            io=io)

    # 7. Setup Logging
    timestamp = datetime.now().strftime("%Y-%m-%d")
    timestamp_detailed = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = config.project_root / "logs" / f"log_{timestamp}" / f"log_{timestamp_detailed}.log"
    setup_logging(log_path, config.log_level, config.debug_mode)
    
    logger = logging.getLogger(__name__)
    logger.info("=== STARTING EXCHANGE ANALYSIS PIPELINE ===")
    logger.info(logger_info_msg)

    # ==========================================
    # PIPELINE EXECUTION
    # ==========================================

    # --- PHASE 1: DOWNLOAD ---
    if config.run_phases["download"]:
        logger.info(f"=== STARTING DOWNLOAD ({config.start} to {config.end}) ===")
        client = EntsoeFileClientAdapter(debug= False, target_zones= config.target_zones, year = year)

        download_generation_demand(client, config)
        download_flows(client, config,"commercial", dayahead=False)
        download_flows(client, config, "commercial", dayahead=True)
        download_flows(client, config, "physical")

       # fetch_simple_metrics(client, config)

        config.io.push_raw_data_to_db(config)

    # --- PHASE 2: PROCESS ---
    gen_data, final_comm, final_phys = None, None, None
    if config.run_phases["process"]:
        logger.info("\n=== STARTING PROCESSING ===")
        
        # A. Generation & Demand
        gen_data = process_generation_demand(config)
        
        # B. Commercial Flows (Total)
        raw_comm = process_flows(config, "commercial", dayahead=False)
        final_comm = balance_flows_symmetry(raw_comm, config, "commercial", dayahead=False)
        
        # C. Day Ahead Flows
        raw_da = process_flows(config,"commercial", dayahead=True)
        balance_flows_symmetry(raw_da, config, "commercial", dayahead=True)
        
        # D. Physical Flows
        raw_phys = process_flows(config, "physical")
        final_phys = balance_flows_symmetry(raw_phys, config, "physical")

      config.io.push_processed_data_to_db(config)

    if save_pickle:
        with open("ioHandler.pkl", "wb") as f:
            pickle.dump(config.io, f)



    # --- PHASE 3: ANALYSIS ---
    if config.run_phases["analysis"]:
        logger.info("\n=== SYNCHRONIZING TIME INDEX ===")
        if not _sync_time_index(config, gen_data, final_comm, final_phys):
            logger.warning("Alle Analyse-Module werden aufgrund fehlender Zeitüberschneidung übersprungen.")
        else:
            logger.info("\n=== STARTING ANALYSIS ===")
            if config.analysis_flags["zone_to_gen_type_analysis"]:
                perform_decomposition_analysis(config, gen_dfs=gen_data, comm_dfs=final_comm)
            if config.analysis_flags["ac_flow_tracing_analysis"]:
               perform_aggregated_flow_tracing(config, gen_dfs=gen_data, phys_flow_dfs=final_phys)
            if config.analysis_flags["dc_flow_tracing_analysis"]:
                perform_direct_flow_tracing(config, gen_dfs=gen_data, phys_flow_dfs=final_phys)
            if config.analysis_flags["pooling_analysis"]:
                perform_pooling_analysis(config, gen_dfs=gen_data, comm_dfs=final_comm, phys_flow_dfs=final_phys)

    config.io.push_analysis_data(config)
    
    if config.run_phases["post_processing"]:
        perform_post_processing_aggregation(config)

if __name__ == "__main__":
    main(year = 2024)