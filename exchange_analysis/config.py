"""
Project: European Electricity Exchange Analysis
Author: Tiernan Buckley
Year: 2026
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Source: https://github.com/INATECH-CIG/exchange_analysis

Description:
Central Configuration Class for the Exchange Analysis Pipeline.
Manages temporal boundaries, execution flags, I/O routing, and spatial topology constraints.
"""

import yaml
import pandas as pd
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine

# Resolve project root explicitly to guarantee localized environment variable discovery
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- API CREDENTIALS ---
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

if not ENTSOE_API_KEY:
    raise EnvironmentError(
        "CRITICAL: ENTSOE_API_KEY not found in .env file. "
        "Please visit https://transparency.entsoe.eu/ to obtain a key."
    )

# --- DATABASE SETTINGS ---
DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")

# --- LOGGING & DEBUG ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

logger = logging.getLogger(__name__)

def get_db_engine():
    """Constructs the SQLAlchemy database engine instance."""
    if not all([DB_USER, DB_PASS, DB_NAME]):
        raise EnvironmentError("Missing critical DB credentials in .env")
        
    encoded_pass = quote_plus(DB_PASS)
    uri = f"postgresql://{DB_USER}:{encoded_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    return create_engine(uri, echo=DEBUG_MODE)

class PipelineConfig:
    def __init__(
        self, 
        date_range: Tuple[str, str],
        key_file: str = "keys.yaml", 
        run_flags: Optional[Dict[str, bool]] = None,
        target_zones: Optional[List[str]] = None,
        data_types: Optional[Dict[str, bool]] = None,
        io_settings: Optional[Dict[str, Any]] = None,
        analysis_flags: Optional[Dict[str, bool]] = None
    ):
        # ==========================================
        # DIRECTORY MAPPING
        # ==========================================
        self.project_root = PROJECT_ROOT
        self.output_dir = self.project_root / "outputs"
        self.input_dir = self.project_root / "inputs"
        
        # ==========================================
        # TEMPORAL BOUNDARIES
        # ==========================================
        self.start = pd.Timestamp(date_range[0], tz="UTC")
        raw_end = pd.Timestamp(date_range[1], tz="UTC")

        self.log_level = LOG_LEVEL
        self.debug_mode = DEBUG_MODE
        
        # Shift absolute midnight bounds backward to ensure API block inclusivity
        if raw_end.hour == 0 and raw_end.minute == 0:
            self.end = raw_end - pd.Timedelta(minutes=1)
            logger.info(f"[Config] Adjusted end date from {raw_end} to {self.end} for inclusive indexing.")
        else:
            self.end = raw_end
        
        self.time_index = pd.date_range(start=self.start, end=self.end, freq="1h")
        self.year = self.start.year 
        
        # ==========================================
        # PIPELINE ORCHESTRATION FLAGS
        # ==========================================
        self.run_phases = {
            "download": True, 
            "process": True, 
            "analysis": True, 
            "post_processing": True
        }
        if run_flags: self.run_phases.update(run_flags)

        self.analysis_flags = {
            "zone_to_gen_type_analysis": True,
            "ac_flow_tracing_analysis": True,
            "dc_flow_tracing_analysis": True,
            "pooling_analysis": True,
        }
        if analysis_flags: self.analysis_flags.update(analysis_flags)

        # ==========================================
        # DATA I/O ROUTING
        # ==========================================
        self.save_csv = True
        self.save_db = True
        self.load_source = 'csv' # Authorized flags: 'csv' or 'db'
        
        if io_settings:
            self.save_csv = io_settings.get("save_csv", self.save_csv)
            self.save_db = io_settings.get("save_db", self.save_db)
            self.load_source = io_settings.get("load_source", self.load_source)

        # ==========================================
        # API DOWNLOAD FILTERS
        # ==========================================
        self.data_types = {
            "generation": True,
            "flows_commercial_total": True,
            "flows_commercial_dayahead": True,
            "flows_physical": True,
            "metrics": True
        }
        if data_types: self.data_types.update(data_types)

        # ==========================================
        # SPATIAL CONFIGURATION (ZONES & TOPOLOGY)
        # ==========================================
        yaml_path = self.project_root / "config.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config YAML not found at {yaml_path}")
            
        with open(yaml_path, 'r') as f:
            topology_data = yaml.safe_load(f)
            raw_neighbours = topology_data.get('neighbours', {})
            self.hvdc_borders = topology_data.get('hvdc_borders', [])
            self.valid_zero_zones = topology_data.get('valid_zero_zones', [])
            zones_to_remove = topology_data.get('zones_to_remove', [])

        # Filter topological map strictly through dictionary comprehension
        self.neighbours_map = {
            k: [v for v in neighbors if v not in zones_to_remove] 
            for k, neighbors in raw_neighbours.items() 
            if k not in zones_to_remove
        }
            
        self.all_zones = list(self.neighbours_map.keys())

        if target_zones:
            self.target_zones = [z for z in target_zones if z in self.all_zones]
            logger.info(f"Configured for subset of zones: {self.target_zones}")
        else:
            self.target_zones = self.all_zones

        # ==========================================
        # CREDENTIALS & METADATA
        # ==========================================
        self.api_key = ENTSOE_API_KEY
            
        self.gen_types_df = pd.read_csv(
            self.input_dir / "generation_data/gen_types_and_emission_factors.csv"
        )
        self.gen_types_list = self.gen_types_df["entsoe"].tolist()

        # Establishes a localized vintage tag for current instantiation
        self.analysis_source_date = pd.Timestamp.utcnow().strftime('%Y-%m-%d')

    @property
    def zones(self):
        """Returns the active list of topologically valid bidding zones."""
        return self.all_zones

    # ==========================================
    # PATH GENERATORS
    # ==========================================
    def get_output_path(self, subfolder: str) -> Path:
        """Constructs and guarantees the existence of temporal-partitioned output directories."""
        path = self.output_dir / subfolder / str(self.year)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_gaps_path(self, subfolder: str) -> Path:
        """Constructs and guarantees the existence of directories for data quality audit logs."""
        path = self.output_dir / subfolder / str(self.year) / "gaps"
        path.mkdir(parents=True, exist_ok=True)
        return path
