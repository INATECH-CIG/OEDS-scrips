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

import sys
from datetime import datetime
from entsoe import EntsoePandasClient
from config import PipelineConfig
from utils import DataIO, setup_logging
import logging

# --- MODULE IMPORTS ---
from download_data import (
    download_generation_demand,
    process_generation_demand,
    download_flows,
    process_flows,
    balance_flows_symmetry,
    fetch_simple_metrics
)
from data_analysis import (
    perform_decomposition_analysis, 
    perform_aggregated_flow_tracing,
    perform_direct_flow_tracing,
    perform_pooling_analysis,
    perform_post_processing_aggregation
)

def main():
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
    
    # 2. Define Period (UTC)
    period = ("2026-01-01 00:00", "2026-01-31 23:59") 

    # 3. Define I/O Settings (Storage & Loading)
    my_io_settings = {
        "save_csv": True,      # Save outputs locally as CSVs
        "save_db": False,        # Push outputs to the TimescaleDB server
        "load_source": "csv"     # 'csv' or 'db'
    }

    # 4. Optional: Download only Subsets of Data
    # -------------------------------------------------------
    # selected_bzs = ["DE_LU", "FR", "BE"]  
    #
    # selected_data_types = {
    #     "generation": True,
    #     "flows_commercial_total": True,
    #     "flows_commercial_dayahead": True, 
    #     "flows_physical": True,
    #     "metrics": False
    # }
    # -------------------------------------------------------

    # 5. Initialize Config
    config = PipelineConfig(
        date_range=period,
        run_flags=my_run_flags,
        io_settings=my_io_settings,
        analysis_flags=analysis_subset,

        # Pass the optional variables here (Uncomment the variables in Step 4 to use them)
        # target_zones=selected_bzs,
        # data_types=selected_data_types
    )
    
    # 6. Initialize IO (Dependency Injection)
    # This checks .env and initializes the DB engine if needed.
    io = DataIO(config)

    # 7. Setup Logging
    timestamp = datetime.now().strftime("%Y-%m-%d")
    timestamp_detailed = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = config.project_root / "logs" / f"log_{timestamp}" / f"log_{timestamp_detailed}.log"
    setup_logging(log_path, config.log_level, config.debug_mode)
    
    logger = logging.getLogger(__name__)
    logger.info("=== STARTING EXCHANGE ANALYSIS PIPELINE ===")

    # ==========================================
    # PIPELINE EXECUTION
    # ==========================================

    # --- PHASE 1: DOWNLOAD ---
    if config.run_phases["download"]:
        logger.info(f"=== STARTING DOWNLOAD ({config.start} to {config.end}) ===")
        client = EntsoePandasClient(api_key=config.api_key)
        
        download_generation_demand(client, config, io)
        download_flows(client, config, io, "commercial", dayahead=False)
        download_flows(client, config, io, "commercial", dayahead=True)
        download_flows(client, config, io, "physical")
        fetch_simple_metrics(client, config, io)

    # --- PHASE 2: PROCESS ---
    gen_data, final_comm, final_phys = None, None, None
    if config.run_phases["process"]:
        logger.info("\n=== STARTING PROCESSING ===")
        
        # A. Generation & Demand
        gen_data = process_generation_demand(config, io)
        
        # B. Commercial Flows (Total)
        raw_comm = process_flows(config, io, "commercial", dayahead=False)
        final_comm = balance_flows_symmetry(raw_comm, config, io, "commercial", dayahead=False)
        
        # C. Day Ahead Flows
        raw_da = process_flows(config, io, "commercial", dayahead=True)
        balance_flows_symmetry(raw_da, config, io, "commercial", dayahead=True)
        
        # D. Physical Flows
        raw_phys = process_flows(config, io, "physical")
        final_phys = balance_flows_symmetry(raw_phys, config, io, "physical")

    # --- PHASE 3: ANALYSIS ---
    if config.run_phases["analysis"]:
        logger.info("\n=== STARTING ANALYSIS ===")

        if config.analysis_flags["zone_to_gen_type_analysis"]:
            perform_decomposition_analysis(config, io, gen_dfs=gen_data, comm_dfs=final_comm)
        
        if config.analysis_flags["ac_flow_tracing_analysis"]:
            perform_aggregated_flow_tracing(config, io, gen_dfs=gen_data, phys_flow_dfs=final_phys)

        if config.analysis_flags["dc_flow_tracing_analysis"]:
            perform_direct_flow_tracing(config, io, gen_dfs=gen_data, phys_flow_dfs=final_phys)
        
        if config.analysis_flags["pooling_analysis"]:
            perform_pooling_analysis(config, io, gen_dfs=gen_data, comm_dfs=final_comm, phys_flow_dfs=final_phys)
    
    if config.run_phases["post_processing"]:
        perform_post_processing_aggregation(config, io)

if __name__ == "__main__":
    main()