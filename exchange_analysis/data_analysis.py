"""
Project: European Electricity Exchange Analysis
Author: Tiernan Buckley
Year: 2026
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Source: https://github.com/INATECH-CIG/exchange_analysis

Description:
Executes the core mathematical logic, decomposing border flows based on generation mixes, 
flow tracing, and performing copper-plate pooling calculations across the European grid.
"""

import pandas as pd
import numpy as np
import os
import tqdm
import logging
from typing import Dict, Optional, Tuple, List, Any
from config import PipelineConfig
from utils import DataIO

logger = logging.getLogger(__name__)

# ==========================================
# LOADER
# ==========================================
def _load_if_missing(
    config: PipelineConfig, 
    io: DataIO, 
    gen_dfs: Optional[Dict[str, pd.DataFrame]], 
    comm_dfs: Optional[Dict[str, pd.DataFrame]] = None, 
    phys_dfs: Optional[Dict[str, pd.DataFrame]] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Lazy-loads generation, commercial, and physical datasets into memory.
    Synchronizes data vintages and enforces standard temporal resampling.
    """
    if gen_dfs is None:
        logger.info("Loading Gen data...")
        gen_dir = config.get_output_path("generation_demand_data_bidding_zones")
        gen_dfs = {}
        for bz in config.zones:
            df = io.load(gen_dir / f"{bz}_generation_demand_data_bidding_zones.csv", "processed_generation", config, bz=bz)
            if df is not None:
                # Extract and synchronize global data vintage
                if "source_download_date" in df.columns and not hasattr(config, "analysis_source_date"):
                    config.analysis_source_date = str(df["source_download_date"].iloc[0]).split()[0]
                # Isolate numeric features during temporal resampling
                gen_dfs[bz] = df.resample("1h").mean(numeric_only=True).fillna(0)
    else:
        logger.info("Gen data already loaded...")
    
    if gen_dfs is None: gen_dfs = {}

    if comm_dfs is None:
        logger.info("Loading Commercial Flow data...")
        comm_dir = config.get_output_path("comm_flow_total_bidding_zones")
        comm_dfs = {}
        for bz in config.zones:
            df = io.load(comm_dir / f"{bz}_comm_flow_total_bidding_zones.csv", "processed_commercial_flows", config, bz=bz)
            if df is not None:
                if "source_download_date" in df.columns and not hasattr(config, "analysis_source_date"):
                    config.analysis_source_date = str(df["source_download_date"].iloc[0]).split()[0]
                comm_dfs[bz] = df.resample("1h").mean(numeric_only=True).fillna(0)
    else:
        logger.info("Commercial Flow data already loaded...")
            
    if comm_dfs is None: comm_dfs = {}

    if phys_dfs is None:
        logger.info("Loading Physical Flow data...")
        flow_dir = config.get_output_path("physical_flow_data_bidding_zones")
        phys_dfs = {}
        for bz in config.zones:
            df = io.load(flow_dir / f"{bz}_physical_flow_data_bidding_zones.csv", "processed_physical_flows", config, bz=bz)
            if df is not None:
                if "source_download_date" in df.columns and not hasattr(config, "analysis_source_date"):
                    config.analysis_source_date = str(df["source_download_date"].iloc[0]).split()[0]
                phys_dfs[bz] = df.resample("1h").mean(numeric_only=True).fillna(0)
    else:
        logger.info("Physical Flow data already loaded...")

    if phys_dfs is None: phys_dfs = {}
    
    return gen_dfs, comm_dfs, phys_dfs

# ==========================================
# PART 1: NEIGHBOR DECOMPOSITION
# ==========================================
def perform_decomposition_analysis(
    config: PipelineConfig, 
    io: DataIO, 
    gen_dfs: Optional[Dict[str, pd.DataFrame]] = None, 
    comm_dfs: Optional[Dict[str, pd.DataFrame]] = None
) -> None:
    """
    Decomposes incoming commercial flows by matching them against the generation 
    mix fractions of the respective source bidding zones.
    """
    logger.info("=== STARTING COMMERCIAL FLOW DECOMPOSITION ===")
    gen_dfs_loaded, comm_dfs_loaded, _ = _load_if_missing(config, io, gen_dfs, comm_dfs=comm_dfs)
    agg_map = config.gen_types_df.groupby(['converted'])['entsoe'].apply(list).to_dict()
    base_out = config.output_dir / "comm_flow_total_bidding_zones" / str(config.year) / "results"
    
    # Initialize output directory structure
    dirs = {t: base_out / t for t in ["per_type_per_bidding_zone", "per_type", "per_agg_type", "per_bidding_zone"]}
    for k in list(dirs.keys()): dirs[f"netted_{k}"] = base_out / f"netted_{k}"

    logger.info("[Decomposition] Preparing Netted Import columns...")
    for bz, df in comm_dfs_loaded.items():
        for col in [col for col in df.columns if "net_export" in col and "Net Export" != col]:
            target_col = col.replace("_net_export", "_netted_import")
            # Derive netted imports by isolating negative net exports
            if target_col not in df.columns: df[target_col] = df[col].clip(upper=0).abs()
                
    logger.info("[Decomposition] Saving raw flow totals (per_bidding_zone)...")
    for bz in comm_dfs_loaded:
        raw_total_imports = pd.DataFrame(index=config.time_index)
        raw_netted_imports = pd.DataFrame(index=config.time_index)
        for n in config.zones:
            if f"{n}_{bz}" in comm_dfs_loaded[bz].columns: raw_total_imports[n] = comm_dfs_loaded[bz][f"{n}_{bz}"]
            if f"{bz}_{n}_netted_import" in comm_dfs_loaded[bz].columns: raw_netted_imports[n] = comm_dfs_loaded[bz][f"{bz}_{n}_netted_import"]

        io.save(raw_total_imports, dirs["per_bidding_zone"] / f"{bz}_import_comm_flow_total_per_bidding_zone.csv", "analysis_cft_total_bz", config, bz=bz)
        io.save(raw_netted_imports, dirs["netted_per_bidding_zone"] / f"{bz}_import_comm_flow_total_netted_per_bidding_zone.csv", "analysis_cft_netted_bz", config, bz=bz)
            
    logger.info("[Decomposition] Calculating generation mix fractions...")
    gen_fractions: Dict[str, pd.DataFrame] = {}
    for bz, df in gen_dfs_loaded.items():
        total = df["Total Generation"].replace(0, 1)
        fracs = df[[c for c in df.columns if c in config.gen_types_list]].div(total, axis=0)
        if "Storage Discharge" in df.columns: fracs["Storage"] = df["Storage Discharge"] / total
        gen_fractions[bz] = fracs

    valid_zones = [z for z in config.zones if z in comm_dfs_loaded]
    logger.info(f"[Decomposition] Processing {len(valid_zones)} zones...")

    for i, bz in enumerate(valid_zones):
        if i % 5 == 0: logger.info(f"   -> Processing {bz}...")
        total_imp_list: List[pd.DataFrame] = []
        netted_imp_list: List[pd.DataFrame] = []
        
        for n in config.zones:
            if f"{n}_{bz}" in comm_dfs_loaded[bz].columns:
                if n in gen_fractions:
                    # Apply source generation fractions to bilateral flow volumes
                    t_df = gen_fractions[n].mul(comm_dfs_loaded[bz][f"{n}_{bz}"], axis=0)
                    t_df.columns = pd.Index([f"{n}_{c}" for c in t_df.columns])
                    total_imp_list.append(t_df)
                    
                    if f"{bz}_{n}_netted_import" in comm_dfs_loaded[bz].columns:
                        n_df = gen_fractions[n].mul(comm_dfs_loaded[bz][f"{bz}_{n}_netted_import"], axis=0)
                        n_df.columns = pd.Index([f"{n}_{c}" for c in n_df.columns])
                        netted_imp_list.append(n_df)
                else:
                    if comm_dfs_loaded[bz][f"{n}_{bz}"].sum() > 0:
                        logger.warning(f"      [Warning] Flow exists from {n}->{bz}, but no generation data for {n}. Skipping.")

        if total_imp_list:
            total_full = pd.concat(total_imp_list, axis=1)
            netted_full = pd.concat(netted_imp_list, axis=1)
            io.save(total_full, dirs["per_type_per_bidding_zone"] / f"{bz}_import_comm_flow_total_per_type_per_bidding_zone.csv", "analysis_cft_total_type_bz", config, bz=bz)
            io.save(netted_full, dirs["netted_per_type_per_bidding_zone"] / f"{bz}_import_comm_flow_total_netted_per_type_per_bidding_zone.csv", "analysis_cft_netted_type_bz", config, bz=bz)
            
            # Aggregate by specific technology type
            per_type = pd.DataFrame(index=config.time_index)
            per_type_net = pd.DataFrame(index=config.time_index)
            for tech in config.gen_types_list:
                cols_exact = [c for c in total_full.columns if c.split('_')[-1].strip() == tech]
                if cols_exact:
                    per_type[tech], per_type_net[tech] = total_full[cols_exact].sum(axis=1), netted_full[cols_exact].sum(axis=1)

            io.save(per_type, dirs["per_type"] / f"{bz}_import_comm_flow_total_per_type.csv", "analysis_cft_total_type", config, bz=bz)
            io.save(per_type_net, dirs["netted_per_type"] / f"{bz}_import_comm_flow_total_netted_per_type.csv", "analysis_cft_netted_type", config, bz=bz)
            
            # Aggregate by broader categorical mappings (e.g., Fossil, Renewable)
            per_agg = pd.DataFrame(index=config.time_index)
            per_agg_net = pd.DataFrame(index=config.time_index)
            for cat, techs in agg_map.items():
                valid = [t for t in techs if t in per_type.columns]
                if valid: per_agg[cat], per_agg_net[cat] = per_type[valid].sum(axis=1), per_type_net[valid].sum(axis=1)
            
            io.save(per_agg, dirs["per_agg_type"] / f"{bz}_import_comm_flow_total_per_agg_type.csv", "analysis_cft_total_agg", config, bz=bz)
            io.save(per_agg_net, dirs["netted_per_agg_type"] / f"{bz}_import_comm_flow_total_netted_per_agg_type.csv", "analysis_cft_netted_agg", config, bz=bz)
            
    logger.info("[Decomposition] Analysis Complete.")

# ==========================================
# PART 2 SHARED: TRACING SAVER
# ==========================================
def _decompose_and_save(
    config: PipelineConfig, 
    io: DataIO, 
    traced_dfs: Dict[str, pd.DataFrame], 
    base_dir: Any, 
    label: str, 
    gen_dfs: Dict[str, pd.DataFrame]
) -> None:
    """Provides unified logic to format, decompose, and persist structural tracing matrices."""
    logger.info(f"[{label.upper()}] Saving and Decomposing Traced Flows...")
    logger.info(f"   -> Output Directory: {base_dir}")
    per_bz_dir = base_dir / "per_bidding_zone"
    per_type_dir = base_dir / "per_bidding_zone_per_type"
    per_type_total_dir = base_dir / "per_type"
    per_agg_total_dir = base_dir / "per_agg_type"
    
    agg_map = config.gen_types_df.groupby(['converted'])['entsoe'].apply(list).to_dict()
    gen_fractions: Dict[str, pd.DataFrame] = {}
    
    logger.info(f"   -> Calculating source generation fractions...")
    for bz, df in gen_dfs.items():
        total = df["Total Generation"].replace(0, 1)
        fracs = df[[c for c in df.columns if c in config.gen_types_list]].div(total, axis=0) 
        if "Storage Discharge" in df.columns: fracs["Storage"] = df["Storage Discharge"] / total
        gen_fractions[bz] = fracs

    total_zones = len(config.zones)
    count = 0
    
    for bz in config.zones:
        if bz not in traced_dfs or traced_dfs[bz].empty: continue
        
        count += 1
        if count % 5 == 0 or count == total_zones:
            logger.info(f"      [{count}/{total_zones}] Saving results for {bz}...")
            
        io.save(traced_dfs[bz], per_bz_dir / f"{bz}_import_flow_tracing_{label}_per_bidding_zone.csv", f"tracing_{label}_bz", config, bz=bz)
        
        type_dfs: List[pd.DataFrame] = []
        for n in config.zones:
            if n != bz and n in traced_dfs[bz].columns and n in gen_fractions:
                t_df = gen_fractions[n].mul(traced_dfs[bz][n], axis=0)
                t_df.columns = pd.Index([f"{n}_{c}" for c in t_df.columns])
                type_dfs.append(t_df)
        
        if type_dfs:
            full_type = pd.concat(type_dfs, axis=1)
            io.save(full_type, per_type_dir / f"{bz}_import_flow_tracing_{label}_per_type_per_bidding_zone.csv", f"tracing_{label}_type_bz", config, bz=bz)
            
            per_type = pd.DataFrame(index=config.time_index)
            for tech in config.gen_types_list:
                cols_exact = [c for c in full_type.columns if c.split('_')[-1].strip() == tech]
                if cols_exact: per_type[tech] = full_type[cols_exact].sum(axis=1)
            io.save(per_type, per_type_total_dir / f"{bz}_import_flow_tracing_{label}_per_type.csv", f"tracing_{label}_type", config, bz=bz)
            
            per_agg = pd.DataFrame(index=config.time_index)
            for cat, techs in agg_map.items():
                valid = [t for t in techs if t in per_type.columns]
                if valid: per_agg[cat] = per_type[valid].sum(axis=1)
            io.save(per_agg, per_agg_total_dir / f"{bz}_import_flow_tracing_{label}_per_agg_type.csv", f"tracing_{label}_agg", config, bz=bz)
            
    logger.info(f"[{label.upper()}] Save Complete.")

# ==========================================
# PART 2A & 2B: FLOW TRACING
# ==========================================
def perform_aggregated_flow_tracing(
    config: PipelineConfig, 
    io: DataIO,
    gen_dfs: Optional[Dict[str, pd.DataFrame]] = None, 
    phys_flow_dfs: Optional[Dict[str, pd.DataFrame]] = None
) -> None:
    """
    Constructs and inverts the grid topology matrix to perform Aggregated Coupling Flow Tracing.
    Net positions define the nodal injections.
    """
    logger.info("Starting Aggregated Coupling Flow Tracing...")
    gen_dfs_loaded, _, phys_flow_dfs_loaded = _load_if_missing(config, io, gen_dfs, phys_dfs=phys_flow_dfs)
    for bz in config.zones:
        if bz in gen_dfs_loaded: gen_dfs_loaded[bz] = gen_dfs_loaded[bz].resample("1h").mean(numeric_only=True).fillna(0)
        if bz in phys_flow_dfs_loaded: phys_flow_dfs_loaded[bz] = phys_flow_dfs_loaded[bz].resample("1h").mean(numeric_only=True).fillna(0)

    agg_tracing = {bz: pd.DataFrame(0.0, index=config.time_index, columns=config.zones, dtype=float) for bz in config.zones}
    agg_dir = config.output_dir / "import_flow_tracing_bidding_zones/agg_coupling" / str(config.year)
    sing_times: List[pd.Timestamp] = []
    
    logger.info("[Agg. Coupling] Inverting Matrices...")
    for t in tqdm.tqdm(config.time_index):
        Pin: List[List[float]] = []
        A: List[List[float]] = []
        net_imps: List[float] = []
        
        # Construct the characteristic matrix (A) and nodal input vector (Pin)
        for bz in config.zones:
            Pin_arr: List[float] = [0.0] * len(config.zones)
            A_arr: List[float] = [0.0] * len(config.zones)
            exports: float = 0.0
            
            for n in [x for x in config.neighbours_map[bz] if x in config.zones and f"{bz}_{x}_net_export" in phys_flow_dfs_loaded[bz].columns]:
                val = phys_flow_dfs_loaded[bz].at[t, f"{bz}_{n}_net_export"]
                if val < 0: A_arr[config.zones.index(n)] = float(val)
                else: exports += float(val)
            
            net_exp = float(phys_flow_dfs_loaded[bz].at[t, "Net Export"])
            if net_exp > 0:
                Pin_arr[config.zones.index(bz)], A_arr[config.zones.index(bz)] = net_exp, exports
                net_imps.append(0.0)
            else:
                A_arr[config.zones.index(bz)] = exports + abs(net_exp)
                net_imps.append(-net_exp)
            Pin.append(Pin_arr)
            A.append(A_arr)
        
        # Evaluate diagonal elements for singularities and apply mathematical smoothing if required
        for idx, bz_name in enumerate(config.zones):
            if A[idx][idx] == 0:
                # Capture state diagnostics prior to applying inversion patch
                bz_net = phys_flow_dfs_loaded[bz_name].at[t, "Net Export"]
                borders = {
                    f"{bz_name}->{n}": phys_flow_dfs_loaded[bz_name].at[t, f"{bz_name}_{n}_net_export"] 
                    for n in config.neighbours_map[bz_name] 
                    if f"{bz_name}_{n}_net_export" in phys_flow_dfs_loaded[bz_name].columns
                }
                
                logger.warning(
                    f"[Singularity Prevented] Time: {t} | {bz_name} diagonal is 0.0. "
                    f"Net Export: {bz_net:.2f} | Borders: {borders}. "
                    f"Applying 1.0 patch to permit inversion."
                )
                
                # Enforce non-zero diagonal to permit matrix inversion
                A[idx][idx] = 1.0
        
        # Execute topology inversion
        try:
            q = np.dot(np.linalg.inv(A), Pin)
            for x in range(len(config.zones)):
                imps = net_imps[x] * q[x]
                for i in range(len(imps)):
                    if imps[i] > 0 and x != i: 
                        agg_tracing[config.zones[x]].at[t, config.zones[i]] = imps[i]
                        
        except np.linalg.LinAlgError: 
            # Log intractable matrices resulting from perfectly symmetrical loops or grid isolation
            sing_times.append(t)
            logger.error(f"FATAL Singular Matrix at time: {t} despite diagonal patches.")
            
            for x in config.zones:
                agg_tracing[x].loc[t] = np.nan
            
    # Persist unresolvable timepoints for subsequent imputation
    log_path = agg_dir / "incalculable_timepoints/incalculable_timepoints.csv"
    if sing_times:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sing_times, columns=["Timepoints"]).to_csv(log_path, index=False)
    elif log_path.exists():
        log_path.unlink()
        
    _decompose_and_save(config, io, agg_tracing, agg_dir, "agg_coupling", gen_dfs_loaded)

def perform_direct_flow_tracing(
    config: PipelineConfig, 
    io: DataIO,
    gen_dfs: Optional[Dict[str, pd.DataFrame]] = None, 
    phys_flow_dfs: Optional[Dict[str, pd.DataFrame]] = None
) -> None:
    """
    Constructs and inverts the grid topology matrix for Direct Coupling Flow Tracing.
    Uses absolute internal generation and demand as the primary nodal properties.
    """
    logger.info("Starting Direct Coupling Flow Tracing...")
    gen_dfs_loaded, _, phys_flow_dfs_loaded = _load_if_missing(config, io, gen_dfs, phys_dfs=phys_flow_dfs)
    for bz in config.zones:
        if bz in gen_dfs_loaded: gen_dfs_loaded[bz] = gen_dfs_loaded[bz].resample("1h").mean(numeric_only=True).fillna(0)
        if bz in phys_flow_dfs_loaded: phys_flow_dfs_loaded[bz] = phys_flow_dfs_loaded[bz].resample("1h").mean(numeric_only=True).fillna(0)

    dir_tracing = {bz: pd.DataFrame(0.0, index=config.time_index, columns=config.zones, dtype=float) for bz in config.zones}
    direct_dir = config.output_dir / "import_flow_tracing_bidding_zones/direct_coupling" / str(config.year)
    sing_times: List[pd.Timestamp] = []
    
    logger.info("[Direct Coupling] Inverting Matrices...")
    for t in tqdm.tqdm(config.time_index):
        Gin: List[List[float]] = []
        A: List[List[float]] = []
        demands: List[float] = []
        
        # Construct the characteristic matrix (A) and absolute nodal input vector (Gin)
        for bz in config.zones:
            Gin_arr: List[float] = [0.0] * len(config.zones)
            A_arr: List[float] = [0.0] * len(config.zones)
            exports: float = 0.0
            
            # Extract absolute generation and load, defaulting to 0.0 for missing temporal indices
            if t in gen_dfs_loaded[bz].index:
                gen_val = float(gen_dfs_loaded[bz].at[t, "Total Generation"])
                load_val = float(gen_dfs_loaded[bz].at[t, "Total Load"])
            else:
                gen_val = 0.0
                load_val = 0.0
                logger.warning(f"Warning: Missing generation array for {bz} at {t}")

            net_exp = float(phys_flow_dfs_loaded[bz].at[t, "Net Export"])
            
            alt_gen = load_val + net_exp
            alt_demand = gen_val - net_exp
            
            for n in [x for x in config.neighbours_map[bz] if x in config.zones and f"{bz}_{x}_net_export" in phys_flow_dfs_loaded[bz].columns]:
                val = phys_flow_dfs_loaded[bz].at[t, f"{bz}_{n}_net_export"]
                if val < 0: A_arr[config.zones.index(n)] = float(val)
                else: exports += float(val)
            
            if load_val > 0:
                Gin_arr[config.zones.index(bz)], A_arr[config.zones.index(bz)] = (alt_gen if alt_gen > 0 else gen_val), load_val + exports
                demands.append(load_val)
            elif alt_demand > 0:
                Gin_arr[config.zones.index(bz)], A_arr[config.zones.index(bz)] = gen_val, alt_demand + exports
                demands.append(alt_demand)
            else:
                Gin_arr[config.zones.index(bz)], A_arr[config.zones.index(bz)] = exports, exports
                demands.append(0.0)
                
            Gin.append(Gin_arr)
            A.append(A_arr)
            
        # Evaluate diagonal elements for singularities and apply mathematical smoothing if required
        for idx, bz_name in enumerate(config.zones):
            if A[idx][idx] == 0:
                # Capture state diagnostics prior to applying inversion patch
                bz_net = phys_flow_dfs_loaded[bz_name].at[t, "Net Export"]
                borders = {
                    f"{bz_name}->{n}": phys_flow_dfs_loaded[bz_name].at[t, f"{bz_name}_{n}_net_export"] 
                    for n in config.neighbours_map[bz_name] 
                    if f"{bz_name}_{n}_net_export" in phys_flow_dfs_loaded[bz_name].columns
                }
                
                logger.warning(
                    f"[Singularity Prevented] Time: {t} | {bz_name} diagonal is 0.0. "
                    f"Net Export: {bz_net:.2f} | Borders: {borders}. "
                    f"Applying 1.0 patch to permit inversion."
                )
                
                # Enforce non-zero diagonal to permit matrix inversion
                A[idx][idx] = 1.0
        
        # Execute topology inversion
        try:
            q = np.dot(np.linalg.inv(A), Gin)
            for x in range(len(config.zones)):
                imps = demands[x] * q[x]
                for i in range(len(imps)):
                    if imps[i] > 0 and x != i: 
                        dir_tracing[config.zones[x]].at[t, config.zones[i]] = imps[i]
                        
        except np.linalg.LinAlgError: 
            # Log intractable matrices resulting from perfectly symmetrical loops or grid isolation
            sing_times.append(t)
            logger.error(f"FATAL Singular Matrix at time: {t} despite diagonal patches.")
            
            for x in config.zones:
                dir_tracing[x].loc[t] = np.nan
    
    # Persist unresolvable timepoints for subsequent imputation
    log_path = direct_dir / "incalculable_timepoints/incalculable_timepoints.csv"
    if sing_times:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sing_times, columns=["Timepoints"]).to_csv(log_path, index=False)
    elif log_path.exists():
        log_path.unlink()
        
    _decompose_and_save(config, io, dir_tracing, direct_dir, "direct_coupling", gen_dfs_loaded)

# ==========================================
# PART 3: POOLING
# ==========================================
def perform_pooling_analysis(
    config: PipelineConfig, 
    io: DataIO,
    gen_dfs: Optional[Dict[str, pd.DataFrame]] = None, 
    comm_dfs: Optional[Dict[str, pd.DataFrame]] = None, 
    phys_flow_dfs: Optional[Dict[str, pd.DataFrame]] = None
) -> None:
    """
    Executes a 'Copper Plate' pooling analysis where net imports are proportionally 
    sourced from all net exporters in the network, independent of exact grid topology.
    """
    logger.info("=== STARTING POOLING ANALYSIS ===")
    gen_dfs_loaded, comm_dfs_loaded, phys_flow_dfs_loaded = _load_if_missing(config, io, gen_dfs, comm_dfs, phys_flow_dfs)
    
    logger.info("[Pooling] 1/4: Calculating generation fractions for all zones...")
    gen_fractions: Dict[str, pd.DataFrame] = {}
    for i, (bz, df) in enumerate(gen_dfs_loaded.items()):
        if i % 10 == 0: logger.info(f"   -> Processing fractions for {bz}...")
        total = df["Total Generation"].replace(0, 1)
        fracs = df[[c for c in df.columns if c in config.gen_types_list]].div(total, axis=0) 
        if "Storage Discharge" in df.columns: fracs["Storage"] = df["Storage Discharge"] / total
        gen_fractions[bz] = fracs.loc[:, ~fracs.columns.duplicated()].fillna(0.0)
        
    agg_map = config.gen_types_df.groupby(['converted'])['entsoe'].apply(list).to_dict()
    base_pool = config.output_dir / "pooling" / str(config.year)

    def save_pool(pooled_dict: Dict[str, pd.DataFrame], name: str, file_p: str) -> None:
        logger.info(f"   -> Saving results for method: {name}")
        dirs = {k: base_pool / name / k for k in ["per_bidding_zone", "per_type_per_bidding_zone", "per_type", "per_agg_type"]}
        
        count = 0
        total_zones = len(pooled_dict)
        
        for bz, df_imp in pooled_dict.items():
            count += 1
            if count % 5 == 0 or count == total_zones:
                logger.info(f"      [{count}/{total_zones}] Saving {bz}...")
                
            io.save(df_imp, dirs["per_bidding_zone"] / f"{bz}_pooled_{file_p}_per_bidding_zone.csv", f"pool_{name}_bz", config, bz=bz)
            
            type_dfs: List[pd.DataFrame] = []
            for src in [s for s in config.zones if s in df_imp.columns and s in gen_fractions]:
                t = gen_fractions[src].mul(df_imp[src], axis=0)
                t.columns = pd.Index([f"{src}_{c}" for c in t.columns])
                type_dfs.append(t)
            
            if type_dfs:
                full = pd.concat(type_dfs, axis=1)
                io.save(full, dirs["per_type_per_bidding_zone"] / f"{bz}_pooled_{file_p}_per_type_per_bidding_zone.csv", f"pool_{name}_type_bz", config, bz=bz)
                
                per_type = pd.DataFrame(index=config.time_index)
                for tech in config.gen_types_list:
                    cols_exact = [c for c in full.columns if c.split('_')[-1].strip() == tech]
                    if cols_exact: per_type[tech] = full[cols_exact].sum(axis=1)
                io.save(per_type, dirs["per_type"] / f"{bz}_pooled_{file_p}_per_type.csv", f"pool_{name}_type", config, bz=bz)
                
                per_agg = pd.DataFrame(index=config.time_index)
                for cat, techs in agg_map.items():
                    valid = [t for t in techs if t in per_type.columns]
                    if valid: per_agg[cat] = per_type[valid].sum(axis=1)
                io.save(per_agg, dirs["per_agg_type"] / f"{bz}_pooled_{file_p}_per_agg_type.csv", f"pool_{name}_agg", config, bz=bz)

    # Construct system-wide export/import matrices for proportional allocation
    tot_exp = pd.DataFrame(index=config.time_index)
    tot_imp = pd.DataFrame(index=config.time_index)
    net_exp = pd.DataFrame(index=config.time_index)
    net_imp = pd.DataFrame(index=config.time_index)
    p_exp = pd.DataFrame(index=config.time_index)
    p_imp = pd.DataFrame(index=config.time_index)
    
    for bz in config.zones:
        if bz in comm_dfs_loaded:
            links = [c for c in comm_dfs_loaded[bz].columns if "net_export" in c and "Net Export" != c]
            tot_exp[bz], tot_imp[bz] = comm_dfs_loaded[bz][links].clip(lower=0).sum(axis=1), comm_dfs_loaded[bz][links].clip(upper=0).abs().sum(axis=1)
            v = comm_dfs_loaded[bz]["Net Export"]
            net_exp[bz], net_imp[bz] = v.clip(lower=0), v.clip(upper=0).abs()
        if bz in phys_flow_dfs_loaded:
            v_p = phys_flow_dfs_loaded[bz]["Net Export"]
            p_exp[bz], p_imp[bz] = v_p.clip(lower=0), v_p.clip(upper=0).abs()

    logger.info("[Pooling] 2/4: Calculating Commercial Link-Based Mix...")
    save_pool({bz: tot_exp.div(tot_exp.sum(axis=1).replace(0, 1), axis=0).mul(tot_imp[bz], axis=0) for bz in config.zones if bz in tot_imp}, "commercial_link_based", "imports")
    
    logger.info("[Pooling] 3/4: Calculating Commercial Net Position Mix...")
    save_pool({bz: net_exp.div(net_exp.sum(axis=1).replace(0, 1), axis=0).mul(net_imp[bz], axis=0) for bz in config.zones if bz in net_imp}, "commercial_net_pos", "net_imports")
    
    logger.info("[Pooling] 4/4: Calculating Physical Net Position Mix...")
    save_pool({bz: p_exp.div(p_exp.sum(axis=1).replace(0, 1), axis=0).mul(p_imp[bz], axis=0) for bz in config.zones if bz in p_imp}, "physical_net_pos", "net_imports")
    
    logger.info("[Pooling] Analysis Complete.")

# ==========================================
# PART 4: POST PROCESSING AGGREGATION
# ==========================================
def perform_post_processing_aggregation(config: PipelineConfig, io: DataIO) -> None:
    """
    Aggregates high-granularity hourly MW flows into final annualized TWh totals
    for high-level methodology comparison.
    """
    logger.info("Starting Post-Processing Aggregation...")
    base_out, year = config.output_dir, str(config.year)
    paths = {
        "CFT": (base_out / f"comm_flow_total_bidding_zones/{year}/results/per_bidding_zone", "analysis_cft_total_bz", "{bz}_import_comm_flow_total_per_bidding_zone.csv"),
        "Netted CFT": (base_out / f"comm_flow_total_bidding_zones/{year}/results/netted_per_bidding_zone", "analysis_cft_netted_bz", "{bz}_import_comm_flow_total_netted_per_bidding_zone.csv"),
        "Pooled Net CFT": (base_out / f"pooling/{year}/commercial_net_pos/per_bidding_zone", "pool_commercial_net_pos_bz", "{bz}_pooled_net_imports_per_bidding_zone.csv"),
        "Pooled Net Phys.": (base_out / f"pooling/{year}/physical_net_pos/per_bidding_zone", "pool_physical_net_pos_bz", "{bz}_pooled_net_imports_per_bidding_zone.csv"),
        "DC Flow Tracing": (base_out / f"import_flow_tracing_bidding_zones/direct_coupling/{year}/per_bidding_zone", "tracing_direct_coupling_bz", "{bz}_import_flow_tracing_direct_coupling_per_bidding_zone.csv"),
        "AC Flow Tracing": (base_out / f"import_flow_tracing_bidding_zones/agg_coupling/{year}/per_bidding_zone", "tracing_agg_coupling_bz", "{bz}_import_flow_tracing_agg_coupling_per_bidding_zone.csv")
    }
    
    totals_out = base_out / f"annual_totals_per_method/{year}"
    sub_outs = {k: totals_out / v for k,v in [("imp_bz", "import/per_bidding_zone"), ("exp_bz", "export/per_bidding_zone"), ("imp_type", "import/per_type"), ("exp_type", "export/per_type"), ("imp_agg", "import/per_agg_type")]}

    def load_clean(io: DataIO, path: Any, table_prefix: str, bz: str, drop: Optional[str] = None) -> Optional[pd.DataFrame]:
        df = io.load(path, table_prefix, config, bz=bz)
        if df is None: return None
        df = df.loc[:, ~df.columns.duplicated()].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        if drop and drop in df.columns: df = df.drop(columns=[drop])
        return df

    # Ingest log of mathematically singular timepoints for fallback allocation
    missing_times: pd.DatetimeIndex = pd.DatetimeIndex([])
    missing_log_path = base_out / f"import_flow_tracing_bidding_zones/agg_coupling/{year}/incalculable_timepoints/incalculable_timepoints.csv"
    if missing_log_path.exists():
        try:
            missing_df = pd.read_csv(missing_log_path)
            if not missing_df.empty and "Timepoints" in missing_df.columns:
                missing_times = pd.to_datetime(missing_df["Timepoints"], utc=True)
        except pd.errors.EmptyDataError:
            pass

    gen_dir = config.get_output_path("generation_demand_data_bidding_zones")
    gen_fractions: Dict[str, pd.DataFrame] = {}
    for bz in config.zones:
        df = load_clean(io, gen_dir / f"{bz}_generation_demand_data_bidding_zones.csv", "processed_generation", bz)
        if df is not None:
            df = df.resample("1h").mean(numeric_only=True).fillna(0)
            total = df["Total Generation"].replace(0, 1)
            fracs = df[[c for c in df.columns if c in config.gen_types_list]].div(total, axis=0)
            if "Storage Discharge" in df.columns: fracs["Storage"] = df["Storage Discharge"] / total
            gen_fractions[bz] = fracs.loc[:, ~fracs.columns.duplicated()].fillna(0.0)

    imports: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {m: {} for m in paths}
    for bz in config.zones:
        for m, (base_p, prefix, fname_template) in paths.items():
            imports[m][bz] = load_clean(io, base_p / fname_template.format(bz=bz), prefix, bz, drop=bz)
        
        ac = imports["AC Flow Tracing"].get(bz)
        pooled = imports["Pooled Net Phys."].get(bz)
        if ac is not None and len(missing_times) > 0 and pooled is not None:
            intersect = ac.index.intersection(missing_times)
            if not intersect.empty: ac.loc[intersect] = pooled.loc[intersect]
        imports["AC Flow Tracing"][bz] = ac

    target_cols = list(set(config.gen_types_list))
    agg_map = config.gen_types_df.groupby(['converted'])['entsoe'].apply(list).to_dict()

    logger.info("[Aggregation] Calculating Annual Totals...")
    for bz in config.zones:
        res_imp_bz = pd.DataFrame(dtype=float)
        res_exp_bz = pd.DataFrame(dtype=float)
        res_imp_type = pd.DataFrame(dtype=float)
        res_exp_type = pd.DataFrame(dtype=float)
        
        for m in paths:
            df_imp = imports[m].get(bz)
            if df_imp is None or df_imp.empty: continue
            
            res_imp_bz.loc[m, df_imp.sum().index] = df_imp.sum()
            imp_h = pd.DataFrame(0.0, index=config.time_index, columns=target_cols)
            for src in df_imp.columns:
                if src in gen_fractions:
                    mix = gen_fractions[src]
                    common = list(set(mix.columns) & set(imp_h.columns))
                    if common: imp_h[common] += mix[common].mul(df_imp[src], axis=0).fillna(0.0)
            res_imp_type.loc[m, imp_h.columns] = imp_h.sum()

            h_exp = pd.Series(0.0, index=config.time_index)
            for n in config.zones:
                if n == bz: continue
                n_imp = imports[m].get(n)
                if n_imp is not None and bz in n_imp.columns:
                    flow = n_imp[bz].fillna(0.0)
                    res_exp_bz.loc[m, n] = flow.sum()
                    h_exp = h_exp + flow
            
            if bz in gen_fractions:
                decomp = gen_fractions[bz].mul(h_exp, axis=0)
                res_exp_type.loc[m, decomp.columns] = decomp.sum()

        # Convert high-granularity MW arrays into annualized Terawatt-hours (TWh)
        res_imp_bz = res_imp_bz.apply(pd.to_numeric, errors='coerce').fillna(0.0) / 1e6
        res_exp_bz = res_exp_bz.apply(pd.to_numeric, errors='coerce').fillna(0.0) / 1e6
        res_imp_type = res_imp_type.apply(pd.to_numeric, errors='coerce').fillna(0.0) / 1e6
        res_exp_type = res_exp_type.apply(pd.to_numeric, errors='coerce').fillna(0.0) / 1e6
        
        io.save(res_imp_bz.T, sub_outs["imp_bz"] / f"{bz}_annual_totals_import_per_bidding_zone_{year}.csv", "annual_imp_bz", config, bz=bz)
        io.save(res_exp_bz.T, sub_outs["exp_bz"] / f"{bz}_annual_totals_export_per_bidding_zone_{year}.csv", "annual_exp_bz", config, bz=bz)
        io.save(res_imp_type.T, sub_outs["imp_type"] / f"{bz}_annual_totals_import_per_type_{year}.csv", "annual_imp_type", config, bz=bz)
        io.save(res_exp_type.T, sub_outs["exp_type"] / f"{bz}_annual_totals_export_per_type_{year}.csv", "annual_exp_type", config, bz=bz)
        
        res_agg = pd.DataFrame(dtype=float)
        for m in res_imp_type.index:
            for cat, techs in agg_map.items():
                valid = [t for t in techs if t in res_imp_type.columns]
                res_agg.loc[m, cat] = res_imp_type.loc[m, valid].sum()
        
        io.save(res_agg.T, sub_outs["imp_agg"] / f"{bz}_annual_totals_import_per_agg_type_{year}.csv", "annual_imp_agg", config, bz=bz)

    logger.info("Post-Processing Complete.")