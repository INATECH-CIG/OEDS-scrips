"""
Project: European Electricity Exchange Analysis
Author: Tiernan Buckley
Year: 2026
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Source: https://github.com/INATECH-CIG/exchange_analysis

Description:
Manages robust database and CSV file I/O operations, handles system logging,
and executes heuristics-based gap filling for missing or anomalous time-series
data.
"""

import time
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Callable, Union
from postgres_utils import df_to_timescale

logger = logging.getLogger(__name__)

# ==========================================
# GAP AUDITING HELPERS
# ==========================================
def _record_gap_method(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, method: str, col_name: str = "ROW") -> None:
    """Appends the specified imputation methodology to the metadata audit trail for a given temporal range."""
    if "gap_filling_method" not in df.columns:
        df["gap_filling_method"] = "None"
        
    mask = (df.index >= start) & (df.index <= end)
    tagged_method = f"[{col_name}] {method}"
    
    none_mask = mask & (df["gap_filling_method"] == "None")
    df.loc[none_mask, "gap_filling_method"] = tagged_method
    
    exist_mask = mask & (df["gap_filling_method"] != "None")
    
    def append_if_missing(current: str) -> str:
        return current if tagged_method in str(current) else f"{current}, {tagged_method}"
        
    df.loc[exist_mask, "gap_filling_method"] = df.loc[exist_mask, "gap_filling_method"].apply(append_if_missing)

def _merge_gap_methods(df_target: pd.DataFrame, df_source: pd.DataFrame) -> None:
    """Consolidates metadata strings when combining parallel datasets to maintain a unified audit trail."""
    if "gap_filling_method" not in df_source.columns: return
    if "gap_filling_method" not in df_target.columns:
        df_target["gap_filling_method"] = "None"
        
    valid_methods = df_source.loc[(df_source["gap_filling_method"] != "None") & df_source["gap_filling_method"].notna(), "gap_filling_method"]
    
    for t, method in valid_methods.items():
        if t in df_target.index:
            curr = df_target.at[t, "gap_filling_method"]
            if curr == "None":
                df_target.at[t, "gap_filling_method"] = method
            elif method not in str(curr):
                df_target.at[t, "gap_filling_method"] = f"{curr}, {method}"

# ==========================================
# DATA I/O HANDLER
# ==========================================

class IOHandler:
    def __init__(self):
        self._tables = {}

    def save(self, df, tablename, directory, config):
        if df is None:
            logger.info(f"Did not save anything for {tablename} because Dataframe is None")
            return

        df = df.copy()
        df.index.name = "time"

        is_result_table = tablename.startswith(
            ("analysis_", "tracing_", "pool_", "annual_", "processed_")
        )

        if is_result_table:
            date_val = getattr(
                config,
                "analysis_source_date",
                pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
            )
            df["source_download_date"] = date_val
            meta_cols = ["gap_filling_method", "bidding_zone", "source_download_date"]
        else:
            df["download_timestamp"] = pd.Timestamp.utcnow().strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            meta_cols = ["gap_filling_method", "bidding_zone", "download_timestamp"]

        data_cols = [c for c in df.columns if c not in meta_cols]
        present_meta = [c for c in meta_cols if c in df.columns]
        df = df[data_cols + present_meta]

        # IMPORTANT: Only save internally and as CSV, NO direct TimescaleDB push during download/processing
        self._tables[tablename] = df.copy()

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        df.to_csv(directory / f"{tablename}.csv", index=False)

    def load(self, tablename, config):
        if tablename in self._tables:
            df = self._tables[tablename].copy()
            df.index = pd.to_datetime(df.index, utc=True)
            mask = (df.index >= config.start) & (df.index <= config.end)
            return df.loc[mask]
        return None

    def push_raw_data_to_db(self, config):
        """
        Transforms the internally stored raw data and pushes it
        into the new TimescaleDB tables in the configured schema.
        """
        logger.info("Starting transformation and push of raw data to TimescaleDB...")

        schema_name = config.db_schema_name

        # 1. Cross Border Flows Bidding Zones Raw
        self._push_cross_border_flows(config, schema_name, raw=True)

        # 2. Zonal Generation Demand Raw
        self._push_zonal_generation_demand(config, schema_name, raw=True)

        logger.info("Raw data transformation and DB push completed.")

    def push_transformed_data_to_db(self, config):
        """
        Transforms the internally stored processed data and pushes it
        once at the end into the new TimescaleDB tables in the configured schema.
        """
        logger.info("Starting transformation and push to TimescaleDB...")

        schema_name = config.db_schema_name

        # 1. Cross Border Flows Bidding Zones
        self._push_cross_border_flows(config, schema_name, raw=False)

        # 2. Zonal Generation Demand
        self._push_zonal_generation_demand(config, schema_name, raw=False)

        # 3. Market Price Dayahead
        self._push_market_prices(config, schema_name)

        # 4. Net Exports
        self._push_net_results(config)

        logger.info("Transformation and DB push completed.")

    def _push_net_results(self, config) -> None:
        """
        Create and push the ``Net_Exports`` table.
        The table aggregates the net-export values from:
        * Generation/Demand (Net Export)
        * Commercial Flows Dayahead (Net Export)
        * Commercial Flows Total (Net Export)
        * Physical Flows (Net Export)
        * SDAC net position (value column)
        The table is written to the same schema defined in the PipelineConfig.
        """
        logger.info("Creating & pushing Net_Exports table...")
        schema_name = config.db_schema_name

        # Helper: safely extract a column, returning a Series of NaNs when missing
        def _extract(col_df: Optional[pd.DataFrame], col_name: str) -> pd.Series:
            if col_df is None:
                return pd.Series([np.nan] * len(config.time_index), index=config.time_index)
            # Accept both "Net_Export" and "Net Export"
            if col_name in col_df.columns:
                return col_df[col_name]
            alt = col_name.replace('_', ' ')
            if alt in col_df.columns:
                return col_df[alt]
            return pd.Series([np.nan] * len(col_df), index=col_df.index)

        net_chunks = []  # one row per zone per timestamp
        for bz in config.zones:
            # Base row (time + bidding zone)
            base = pd.DataFrame(index=config.time_index)
            base["time"] = config.time_index
            base["bidding_zone"] = bz

            # 1. Generation/Demand Net Export
            gen_df = self._tables.get(f"{bz}_generation_demand")
            base["generation_demand_net_export"] = _extract(gen_df, "Net_Export")

            # 2. Commercial Flows Dayahead Net Export
            comm_da_df = self._tables.get(f"{bz}_comm_flow_dayahead_bidding_zones")
            base["commercial_flows_dayahead_net_export"] = _extract(comm_da_df, "Net_Export")

            # 3. Commercial Flows Total Net Export
            comm_tot_df = self._tables.get(f"{bz}_comm_flow_total_bidding_zones")
            base["commercial_flows_total_net_export"] = _extract(comm_tot_df, "Net_Export")

            # 4. Physical Flows Net Export
            phys_df = self._tables.get(f"{bz}_physical_flow_data_bidding_zones")
            base["physical_flows_net_export"] = _extract(phys_df, "Net_Export")

            # 5. SDAC Net Position (column named "value")
            sdac_df = self._tables.get(f"{bz}_net_positions_dayahead")
            if sdac_df is not None and "Value" in sdac_df.columns:
                base["sdac_net_position"] = sdac_df["Value"]
            else:
                base["sdac_net_position"] = np.nan

            net_chunks.append(base)

        # Concatenate all zones
        net_df = pd.concat(net_chunks, ignore_index=True)

        # Ensure correct column order
        final_order = [
            "time",
            "bidding_zone",
            "generation_demand_net_export",
            "commercial_flows_dayahead_net_export",
            "commercial_flows_total_net_export",
            "physical_flows_net_export",
            "sdac_net_position",
        ]
        net_df = net_df[final_order]

        # Push to TimescaleDB
        df_to_timescale(net_df, "Net_Exports", schema_name)
        logger.info("Net_Exports table successfully pushed.")

    def _push_cross_border_flows(self, config, schema_name, raw=False):
        logger.info(f"Transforming Cross Border Flows {'Raw' if raw else 'Processed'}...")
        flow_chunks = []

        for bz in config.zones:
            # Define keys based on raw flag
            if raw:
                total_key = f"{bz}_raw_commercial_flows"
                da_key = f"{bz}_raw_commercial_flows_dayahead"
                phys_key = f"{bz}_raw_physical_flows"
            else:
                total_key = f"{bz}_comm_flow_total_bidding_zones"
                da_key = f"{bz}_comm_flow_dayahead_bidding_zones"
                phys_key = f"{bz}_physical_flow_data_bidding_zones"

            # Load dataframes
            df_total = self._tables.get(total_key)
            df_da = self._tables.get(da_key)
            df_phys = self._tables.get(phys_key)

            for n in config.neighbours_map.get(bz, []):
                # Define column names
                col = f"{bz}_{n}"
                net_col = f"{col}_net_export"

                # Regular flows (netted = false)
                chunk_reg = pd.DataFrame(index=config.time_index)
                chunk_reg['From Zone'] = bz
                chunk_reg['To Zone'] = n
                chunk_reg['Netted'] = False

                if df_total is not None and col in df_total:
                    chunk_reg['Comm Flow Total'] = df_total[col]
                    chunk_reg['Gap Filling (Comm Flow Total)'] = df_total['gap_filling_method']

                if df_da is not None and col in df_da:
                    chunk_reg['Comm Flow Dayahead'] = df_da[col]
                    chunk_reg['Gap Filling (Comm Flow Dayahead)'] = df_da['gap_filling_method']

                if df_phys is not None and col in df_phys:
                    chunk_reg['Physical Flow'] = df_phys[col]
                    chunk_reg['Gap Filling (Physical Flow)'] = df_phys['gap_filling_method']

                flow_chunks.append(chunk_reg)

                # Netted flows (netted = True) - only applicable for processed data
                if not raw:
                    has_net_data = (df_total is not None and net_col in df_total) or \
                                   (df_da is not None and net_col in df_da) or \
                                   (df_phys is not None and net_col in df_phys)

                    if has_net_data:
                        chunk_net = pd.DataFrame(index=config.time_index)
                        chunk_net['From Zone'] = bz
                        chunk_net['To Zone'] = n
                        chunk_net['Netted'] = True

                        if df_total is not None and net_col in df_total:
                            chunk_net['Comm Flow Total'] = df_total[net_col]
                            chunk_net['Gap Filling (Comm Flow Total)'] = df_total['gap_filling_method']

                        if df_da is not None and net_col in df_da:
                            chunk_net['Comm Flow Dayahead'] = df_da[net_col]
                            chunk_net['Gap Filling (Comm Flow Dayahead)'] = df_da['gap_filling_method']

                        if df_phys is not None and net_col in df_phys:
                            chunk_net['Physical Flow'] = df_phys[net_col]
                            chunk_net['Gap Filling (Physical Flow)'] = df_phys['gap_filling_method']

                        flow_chunks.append(chunk_net)

        if flow_chunks:
            # combine chunks
            final_flows = pd.concat(flow_chunks).reset_index()

            # rename time collumn for timescale
            final_flows = final_flows.rename(columns={'index': 'time'})

            # Updated column order to include the Gap Filling columns
            ordered_cols = [
                'time',
                'From Zone',
                'To Zone',
                'Netted',
                'Comm Flow Total',
                'Comm Flow Dayahead',
                'Physical Flow', 'Gap Filling (Comm Flow Total)',
                'Gap Filling (Comm Flow Dayahead)',
                'Gap Filling (Physical Flow)'
            ]

            if raw:
                # If raw data doesn't have a 'Netted' concept, remove it from the order
                if 'Netted' in ordered_cols:
                    ordered_cols.remove('Netted')

            existing_ordered_cols = [c for c in ordered_cols if c in final_flows.columns]
            final_flows = final_flows[existing_ordered_cols]

            # push to database
            table_name = "Cross_Border_Flows_Bidding_Zones_Raw" if raw else "Cross_Border_Flows_Bidding_Zones"
            df_to_timescale(final_flows, table_name, schema_name)

    def _push_zonal_generation_demand(self, config, schema_name, raw=False):
        logger.info(f"Transforming Zonal Generation Demand {'Raw' if raw else 'Processed'}...")
        gen_chunks = []

        for bz in config.zones:
            if raw:
                df_gen = self._tables.get(f"{bz}_raw_generation")
                df_load = self._tables.get(f"{bz}_raw_load")
                if df_gen is not None or df_load is not None:
                    # Combine generation and load data, drop duplicate columns if they overlap
                    dfs_to_concat = [df for df in [df_gen, df_load] if df is not None]
                    chunk = pd.concat(dfs_to_concat, axis=1)
                    chunk = chunk.loc[:, ~chunk.columns.duplicated()]
                    chunk['zone'] = bz
                    gen_chunks.append(chunk)
            else:
                df_gen = self._tables.get(f"{bz}_generation_demand")
                if df_gen is not None:
                    chunk = df_gen.copy()
                    chunk['zone'] = bz
                    gen_chunks.append(chunk)

        if gen_chunks:
            final_gen = pd.concat(gen_chunks).reset_index()
            final_gen = final_gen.rename(columns={'index': 'time'})
            # Clean up column names (remove spaces and brackets for SQL compatibility)
            final_gen.columns = [c.replace(' ', '_').replace('(', '').replace(')', '') for c in final_gen.columns]

            # 4. Ensure 'zone' is the second column right after 'time'
            cols = list(final_gen.columns)
            if 'zone' in cols:
                cols.remove('zone')
                cols.insert(1, 'zone')
                final_gen = final_gen[cols]


            if not raw:
                # Ensure 'gap_filling_method' and 'download_timestamp' are the last 2 columns
                for col in ['gap_filling_method', 'download_timestamp']:
                    if col in cols:
                        cols.remove(col)
                if 'gap_filling_method' in final_gen.columns:
                    cols.append('gap_filling_method')
                if 'download_timestamp' in final_gen.columns:
                    cols.append('download_timestamp')
                final_gen = final_gen[cols]

            table_name = "Zonal_Generation_Demand_Raw" if raw else "Zonal_Generation_Demand"
            final_gen['time'] = pd.to_datetime(final_gen['time'], utc=True)
            df_to_timescale(final_gen, table_name, schema_name)

    def _push_market_prices(self, config, schema_name):
        logger.info("Transforming Market Price Dayahead...")
        price_chunks = {}

        for bz in config.zones:
            df_price = self._tables.get(f"{bz}_market_price_dayahead")
            if df_price is not None:
                if 'Value' in df_price.columns:
                    price_chunks[bz] = df_price['Value']
                elif not df_price.empty:
                    # Fallback if column name is different
                    price_chunks[bz] = df_price.iloc[:, 0]

        if price_chunks:
            final_price = pd.DataFrame(price_chunks)
            final_price = final_price.reset_index().rename(columns={'index': 'time'})
            df_to_timescale(final_price, "Market_Price_Dayahead", schema_name)


# ==========================================
# LOGGING & API UTILS 
# ==========================================
def setup_logging(log_file_path: Path, log_level_str: str, debug_mode: bool) -> None:
    """Configures standard output and file-based logging streams with conditional formatting."""
    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s' if not debug_mode 
        else '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    if root_logger.hasHandlers(): root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    root_logger.info(f"Logging initialized. Level: {log_level_str} | Debug Mode: {debug_mode}")

def safe_query(func: Callable, max_retries: int = 3, delay: int = 2, context: Optional[str] = None, **kwargs: Any) -> Any:
    """Executes API queries with exponential backoff and localized error handling to mitigate transient failures."""
    for attempt in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            error_msg = str(e)
            if not error_msg.strip(): error_msg = repr(e)
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f" | API Response: {e.response.text}"

            msg = f"[Attempt {attempt + 1}/{max_retries}] Failed"
            if context: msg += f" for {context}"
            msg += f": {error_msg}"
            logger.warning(msg)

            if "NoMatchingDataError" in error_msg: 
                logger.warning(f"Data gap detected for {context}: Source returned empty.")
                return None

            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                logger.error(f"CRITICAL FAILURE: Skipping {context if context else 'query'} after max retries.", exc_info=True)
                return None
    return None

# ==========================================
# GAP FILLING ENGINE
# ==========================================
def default_rules(series: pd.Series, gaps: pd.DataFrame, inferred_freq: pd.Timedelta) -> None:
    """
    Establishes baseline heuristics for time-series imputation based on gap duration and temporal location.
    Applies to both standard missing data ('nan') and filtered outliers ('invalid_data').
    """
    gaps["method"] = "ZERO"
    MAX_WEEK_BEFORE = pd.Timedelta(weeks=1)
    MAX_LINEAR = pd.Timedelta(hours=3)
    
    target_types = ["nan", "invalid_data"]

    gaps.loc[
        (gaps["type"].isin(target_types)) & (gaps["duration"] * inferred_freq <= MAX_WEEK_BEFORE) &
        (gaps["start"] - series.index[0] >= MAX_WEEK_BEFORE), "method",
    ] = "WEEK_BEFORE"

    gaps.loc[
        (gaps["type"].isin(target_types)) & (gaps["duration"] * inferred_freq <= MAX_LINEAR) &
        (gaps["start"] > series.index[0]) & (gaps["end"] < series.index[-1]), "method",
    ] = "LINEAR"

    gaps.loc[
        (gaps["type"].isin(target_types)) & (gaps["duration"] * inferred_freq <= MAX_LINEAR) &
        (gaps["start"] > series.index[0]) & (gaps["end"] == series.index[-1]), "method",
    ] = "FORWARD_FILL"

    gaps.loc[
        (gaps["type"].isin(target_types)) & (gaps["duration"] * inferred_freq <= MAX_LINEAR) &
        (gaps["start"] == series.index[0]) & (gaps["end"] < series.index[-1]), "method",
    ] = "BACKWARD_FILL"

    mask_invalid = gaps["type"] == "invalid_data"
    if mask_invalid.any():
        gaps.loc[mask_invalid, "method"] = "FILTERED_OUTLIER_" + gaps.loc[mask_invalid, "method"]

def fill_gaps_series(series: pd.Series, gaps: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Applies targeted imputation arrays to identified temporal gaps within a continuous 1D series."""
    gaps["success"] = False
    gaps["filled_values"] = 0
    gaps["filled_quantity"] = 0.0

    for i, gap in gaps.iterrows():
        start, end, duration, method = gap["start"], gap["end"], gap["duration"], gap["method"]
        if method == "ZERO":
            series.loc[start:end] = 0
        elif method == "LINEAR":
            pos_start = series.index.get_loc(start)
            series.loc[start:end] = np.linspace(series.iloc[pos_start - 1], series.iloc[pos_start + duration], duration + 2)[1:-1]
        elif method == "FORWARD_FILL":
            series.loc[start:end] = series.iloc[series.index.get_loc(start) - 1]
        elif method == "BACKWARD_FILL":
            series.loc[start:end] = series.iloc[series.index.get_loc(start) + duration]
        elif method == "WEEK_BEFORE":
            one_week = pd.Timedelta(weeks=1)
            series.loc[start:end] = series.loc[(start - one_week):(end - one_week)].values

        gaps.loc[i, "success"] = series.loc[start:end].count() > 0
        gaps.loc[i, "filled_values"] = series.loc[start:end].count()
        gaps.loc[i, "filled_quantity"] = series.loc[start:end].sum()

    return series, gaps

def find_gaps_series(
    series: pd.Series,
    output_dict: Optional[Dict[str, pd.DataFrame]] = None,
    check_negatives: bool = False,
    allow_negatives: Optional[List[str]] = None,
    fill_gaps: bool = False,
    gap_filling_rules: Optional[Callable] = None
) -> pd.Series:
    """Scans a continuous series to isolate, measure, and classify missing or invalid temporal blocks."""
    # Ignore non-numerical metadata features
    if not pd.api.types.is_numeric_dtype(series):
        return series
    
    if allow_negatives is None: allow_negatives = []

    # Identify structural NaNs and physical outliers distinctly
    is_invalid = series >= 100000
    is_nan = series.isna()

    series = series.mask(is_invalid, np.nan)

    def extract_blocks(mask: pd.Series, gap_type: str) -> pd.DataFrame:
        starts = mask & (~mask.shift(1, fill_value=False))
        ends = mask & (~mask.shift(-1, fill_value=False))
        df = pd.DataFrame({"start": series[starts].index, "end": series[ends].index})
        if not df.empty:
            df["duration"] = df.apply(lambda row: mask[row["start"] : row["end"]].sum(), axis=1).astype(int)
        else:
            df["duration"] = pd.Series(dtype=int)
        df["value"] = np.nan
        df["type"] = gap_type
        return df

    gaps = pd.concat([
        extract_blocks(is_nan, "nan"),
        extract_blocks(is_invalid, "invalid_data")
    ], ignore_index=True)

    if check_negatives and (str(series.name) not in allow_negatives):
        is_neg = series < 0
        negs = extract_blocks(is_neg, "negative")
        if not negs.empty:
            negs["value"] = negs.apply(lambda row: series[row["start"] : row["end"]].sum(), axis=1)
        gaps = pd.concat([gaps, negs], ignore_index=True)

    gaps = gaps.sort_values(by="start").reset_index(drop=True)

    inferred_freq = pd.infer_freq(series.index[:3])
    if (inferred_freq is not None) and (len(inferred_freq) == 1): inferred_freq = "1" + inferred_freq
    freq_td = pd.to_timedelta(inferred_freq) if inferred_freq else pd.Timedelta(hours=1)
    gaps["method"] = "UNDEFINED"

    if gap_filling_rules is not None: gap_filling_rules(series, gaps, freq_td)
    if fill_gaps: series, gaps = fill_gaps_series(series, gaps)
    if output_dict is not None: output_dict[str(series.name)] = gaps
    
    return series

def find_gaps(
    df: pd.DataFrame,
    check_negatives: bool = False,
    allow_negatives: Optional[List[str]] = None,
    fill_gaps: bool = False,
    gap_filling_rules: Callable = default_rules
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Iterates row-level gap scanning across a primary DataFrame matrix."""
    if allow_negatives is None: allow_negatives = []
    output_dict: Dict[str, pd.DataFrame] = {}
    df_result = df.apply(find_gaps_series, axis=0, output_dict=output_dict, check_negatives=check_negatives,
                         allow_negatives=allow_negatives, fill_gaps=fill_gaps, gap_filling_rules=gap_filling_rules)
    return df_result, output_dict

def patch_gaps_with_dayahead(
    flow_df: pd.DataFrame,
    gap_dict: Dict[str, pd.DataFrame],
    bz: str,
    neighbour: str,
    config: Any, 
    min_gap_length: pd.Timedelta = pd.Timedelta(weeks=1)
) -> pd.DataFrame:
    """Leverages day-ahead commercial schedules as a physical proxy to impute extended missing flow blocks."""
    long_gaps: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for col in [f"{bz}_{neighbour}", f"{neighbour}_{bz}"]:
        if col in gap_dict:
            for _, row in gap_dict[col].iterrows():
                if (row["end"] - row["start"]) > min_gap_length:
                    long_gaps.append((col, row["start"], row["end"]))

    if not long_gaps:
        return flow_df

    table_name = "processed_commercial_flows_da"

    da_df = config.io.load(f"{bz}_raw_commercial_flows_dayahead", config)

    if da_df is None or da_df.empty:
        return flow_df

    patched_count = 0
    for col, start, end in long_gaps:
        if col in da_df.columns:
            replacement = da_df.loc[start:end, col]

            if not (replacement.empty or replacement.isna().all()):
                flow_df.loc[start:end, col] = replacement
                patched_count += 1
                _record_gap_method(flow_df, start, end, "DAYAHEAD_PROXY", col_name=col)

    if patched_count > 0:
        logger.info(f"   -> [Patch] Used {table_name} to fill {patched_count} long-duration gaps for {bz}.")

    return flow_df

def fill_gaps_wrapper(
    df: pd.DataFrame,
    gaps_dir: Optional[Path],
    prefix: str,
    config: Optional[Any] = None,
    bz: Optional[str] = None,
    flow_type: Optional[str] = None,
    dayahead: bool = False
) -> pd.DataFrame:
    """Orchestrates the detection, rule assignment, and execution of the gap-filling sequence."""
    if df.empty: return df
    
    if "gap_filling_method" not in df.columns:
        df["gap_filling_method"] = "None"
        
    _, gaps_dict = find_gaps(df, check_negatives=False, fill_gaps=False)

    if config and bz and (flow_type == "commercial") and (not dayahead):
        if hasattr(config, 'neighbours_map') and bz in config.neighbours_map:
            for neighbour in [n for n in config.neighbours_map[bz] if f"{bz}_{n}" in df.columns]:
                df = patch_gaps_with_dayahead(df, gaps_dict, bz, neighbour, config)

    df_filled, new_gaps_dict = find_gaps(df, check_negatives=False, fill_gaps=True, gap_filling_rules=default_rules)

    for col_name, gap_df in new_gaps_dict.items():
        if gap_df.empty: continue
        for _, row in gap_df.iterrows():
            if row.get("success", True):
                _record_gap_method(df_filled, row["start"], row["end"], row["method"], col_name=str(col_name))

    if gaps_dir:
        for key, gap_df in new_gaps_dict.items():
            file_path = gaps_dir / f"{prefix}_{str(key).replace('/', '_').replace(' ', '_')}_gaps.csv"
            if not gap_df.empty:
                gap_df.to_csv(file_path)
            else:
                if file_path.exists():
                    file_path.unlink()

    return df_filled

def correct_zero_values(df: pd.DataFrame, gaps_dir: Path, bz: str, config: Any, flow_type: str = "commercial") -> pd.DataFrame:
    """
    Identifies and categorizes mathematically singular states (0 MW) as either 
    systemic dropouts or isolated bilateral failures, applying tiered fallback methodologies.
    Strictly recalculates net positions to preserve arithmetic closure.
    """
    if df.empty: return df
    if "gap_filling_method" not in df.columns: df["gap_filling_method"] = "None"

    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty: return df

    # Isolate physical base flow columns to prevent recursive patching of derived metrics
    base_flow_cols = [c for c in num_df.columns if "Net Export" not in c and "_net_export" not in c]

    # ========================================================
    # PHASE 1: ZERO IDENTIFICATION AND AUDIT LOGGING
    # ========================================================
    if "Total Generation" in df.columns:
        gen_mask = df.get("Total Generation", pd.Series(1, index=df.index)) == 0
        load_mask = df.get("Total Load", df.get("Demand", pd.Series(1, index=df.index))) == 0
        global_zero_mask = gen_mask | load_mask
    else:
        # Evaluate for complete nodal isolation (systemic reporting failure)
        global_zero_mask = (num_df[base_flow_cols] == 0).all(axis=1)
        
        # Exempt defined geographic islands where zero-flow states are physically permissible
        if bz in getattr(config, 'valid_zero_zones', []):
            global_zero_mask = pd.Series(False, index=df.index)

    zeros_df = df[global_zero_mask]
    file_path = gaps_dir / f"{bz}_zeros.csv"

    if not zeros_df.empty:
        zeros_df.to_csv(file_path)
    else:
        if file_path.exists(): file_path.unlink()

    # ========================================================
    # PHASE 2: TIERED PATCHING LOGIC
    # ========================================================
    one_week = pd.Timedelta(weeks=1)

    def apply_patch(condition_mask: pd.Series, cols: list, prefix: str):
        if not condition_mask.any() or not cols: return

        blocks = condition_mask.ne(condition_mask.shift()).cumsum()
        gap_lengths = condition_mask.groupby(blocks).transform('count')
        clean_mean = num_df[cols].replace(0, np.nan).mean().fillna(0)

        for timestamp in df[condition_mask].index:
            success = False
            gap_len = gap_lengths.at[timestamp]

            if gap_len > 24:
                df.loc[timestamp, cols] = clean_mean
                _record_gap_method(df, timestamp, timestamp, f"{prefix}_LONG_GAP_GLOBAL_MEAN", "SYSTEM")
                continue

            if gap_len <= 3:
                try:
                    block_id = blocks.at[timestamp]
                    block_timestamps = df[blocks == block_id].index
                    prev_t, next_t = block_timestamps[0] - pd.Timedelta(hours=1), block_timestamps[-1] + pd.Timedelta(hours=1)
                    
                    if prev_t in df.index and next_t in df.index:
                        val_prev, val_next = df.loc[prev_t, cols], df.loc[next_t, cols]
                        if val_prev.sum() > 0 and val_next.sum() > 0:
                            pos = list(block_timestamps).index(timestamp) + 1
                            step = (val_next - val_prev) / (gap_len + 1)
                            df.loc[timestamp, cols] = val_prev + (step * pos)
                            _record_gap_method(df, timestamp, timestamp, f"{prefix}_LINEAR", "SYSTEM")
                            success = True
                except Exception: pass

            if not success:
                patch_time = timestamp - one_week
                if patch_time < getattr(config, 'start', df.index.min()): patch_time = timestamp + one_week
                if patch_time in df.index:
                    donor = df.loc[patch_time, cols]
                    if donor.sum() > 0:
                        df.loc[timestamp, cols] = donor
                        _record_gap_method(df, timestamp, timestamp, f"{prefix}_WEEK_BEFORE", "SYSTEM")
                        success = True

            if not success:
                df.loc[timestamp, cols] = clean_mean
                _record_gap_method(df, timestamp, timestamp, f"{prefix}_GLOBAL_MEAN_FALLBACK", "SYSTEM")

    # ========================================================
    # PHASE 3: CATEGORICAL IMPUTATION EXECUTION
    # ========================================================
    
    # 3A. Generation Patching
    if "Total Generation" in df.columns:
        gen_cols = [c for c in num_df.columns if c not in ["Demand", "Total Load", "Storage Charge", "Actual Load", "Net Export"] and "net_export" not in c]
        apply_patch(df["Total Generation"] == 0, gen_cols, "GEN_ZERO")

    # 3B. Demand Patching
    target_col = next((col for col in ["Demand", "Total Load"] if col in df.columns), None)
    if target_col:
        dem_cols = [c for c in num_df.columns if c in ["Demand", "Total Load", "Storage Charge"]]
        apply_patch(df[target_col] == 0, dem_cols, "LOAD_ZERO")

    # 3C. Flow Patching
    if "Total Generation" not in df.columns and "Total Load" not in df.columns:
        allowed_islands = getattr(config, 'valid_zero_zones', [])
        
        if bz not in allowed_islands:
            # Scenario A: Commercial Matrix
            if flow_type == "commercial":
                flow_mask = (num_df[base_flow_cols] == 0).all(axis=1)
                apply_patch(flow_mask, base_flow_cols, "COMM_FLOW_SYSTEM_ZERO")
                
            # Scenario B: Physical Matrix
            elif flow_type == "physical":
                # Step 1: System-wide Nodal Checks
                row_zero_mask = (num_df[base_flow_cols] == 0).all(axis=1)
                if row_zero_mask.any():
                    apply_patch(row_zero_mask, base_flow_cols, "PHYS_FLOW_SYSTEM_ZERO")
                
                # Step 2: Isolated Bilateral Checks
                checked_cols = set()
                for col in base_flow_cols:
                    if col in checked_cols: continue
                    
                    parts = col.split('_')
                    if len(parts) >= 2:
                        target = col.replace(f"{bz}_", "") if col.startswith(f"{bz}_") else col.replace(f"_{bz}", "")
                        col_out, col_in = f"{bz}_{target}", f"{target}_{bz}"
                        border_key = "_".join(sorted([bz, target]))
                        
                        if col_out in df.columns and col_in in df.columns:
                            checked_cols.update([col_out, col_in])
                            
                            if border_key in getattr(config, 'hvdc_borders', []): continue 
                            
                            bilateral_zero_mask = (df[col_out] == 0) & (df[col_in] == 0)
                            if bilateral_zero_mask.any():
                                apply_patch(bilateral_zero_mask, [col_out, col_in], f"PHYS_BILATERAL_ZERO_[{target}]")

    # ========================================================
    # PHASE 4: DETERMINISTIC RECALCULATION
    # ========================================================
    # Reconstruct structural net exports following base-layer imputation to enforce topological symmetry
    net_export_cols = [c for c in df.columns if c.endswith("_net_export")]
    if net_export_cols:
        for net_col in net_export_cols:
            col_out = net_col.replace("_net_export", "")
            target = col_out.replace(f"{bz}_", "") if col_out.startswith(f"{bz}_") else col_out.replace(f"_{bz}", "")
            col_in = f"{target}_{bz}"
            
            out_val = df[col_out] if col_out in df.columns else 0.0
            in_val = df[col_in] if col_in in df.columns else 0.0
            
            df[net_col] = out_val - in_val
            
        df["Net Export"] = df[net_export_cols].sum(axis=1)

    return df