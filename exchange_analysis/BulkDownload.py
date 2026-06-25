import pandas as pd
from entsoe.files import EntsoeFileClient

from entsoe.files import EntsoeFileClient
import pandas as pd
from collections import defaultdict
from env import pw

AREAS = {
    "DE_50HZ": ('10YDE-VE-------2', '50Hertz CA, DE(50HzT) BZA', 'Europe/Berlin'),
    "AL": ('10YAL-KESH-----5', 'Albania, OST BZ / CA / MBA', 'Europe/Tirane'),
    "DE_AMPRION": ('10YDE-RWENET---I', 'Amprion CA', 'Europe/Berlin'),
    "AT": ('10YAT-APG------L', 'Austria, APG BZ / CA / MBA', 'Europe/Vienna'),
    "BY": ('10Y1001A1001A51S', 'Belarus BZ / CA / MBA', 'Europe/Minsk'),
    "BE": ('10YBE----------2', 'Belgium, Elia BZ / CA / MBA', 'Europe/Brussels'),
    "BA": ('10YBA-JPCC-----D', 'Bosnia Herzegovina, NOS BiH BZ / CA / MBA', 'Europe/Sarajevo'),
    "BG": ('10YCA-BULGARIA-R', 'Bulgaria, ESO BZ / CA / MBA', 'Europe/Sofia'),
    "CZ_DE_SK": ('10YDOM-CZ-DE-SKK', 'BZ CZ+DE+SK BZ / BZA', 'Europe/Prague'),
    "HR": ('10YHR-HEP------M', 'Croatia, HOPS BZ / CA / MBA', 'Europe/Zagreb'),
    "CWE": ('10YDOM-REGION-1V', 'CWE Region', 'Europe/Brussels'),
    "CY": ('10YCY-1001A0003J', 'Cyprus, Cyprus TSO BZ / CA / MBA', 'Asia/Nicosia'),
    "CZ": ('10YCZ-CEPS-----N', 'Czech Republic, CEPS BZ / CA/ MBA', 'Europe/Prague'),
    "DE_AT_LU": ('10Y1001A1001A63L', 'DE-AT-LU BZ', 'Europe/Berlin'),
    "DE_LU": ('10Y1001A1001A82H', 'DE-LU BZ / MBA', 'Europe/Berlin'),
    "DK": ('10Y1001A1001A65H', 'Denmark', 'Europe/Copenhagen'),
    "DK_1": ('10YDK-1--------W', 'DK1 BZ / MBA', 'Europe/Copenhagen'),
    "DK_1_NO_1": ('46Y000000000007M', 'DK1 NO1 BZ', 'Europe/Copenhagen'),
    "DK_2": ('10YDK-2--------M', 'DK2 BZ / MBA', 'Europe/Copenhagen'),
    "DK_CA": ('10Y1001A1001A796', 'Denmark, Energinet CA', 'Europe/Copenhagen'),
    "EE": ('10Y1001A1001A39I', 'Estonia, Elering BZ / CA / MBA', 'Europe/Tallinn'),
    "FI": ('10YFI-1--------U', 'Finland, Fingrid BZ / CA / MBA', 'Europe/Helsinki'),
    "FR": ('10YFR-RTE------C', 'France, RTE BZ / CA / MBA', 'Europe/Paris'),
    "DE": ('10Y1001A1001A83F', 'Germany', 'Europe/Berlin'),
    "GR": ('10YGR-HTSO-----Y', 'Greece, IPTO BZ / CA/ MBA', 'Europe/Athens'),
    "HU": ('10YHU-MAVIR----U', 'Hungary, MAVIR CA / BZ / MBA', 'Europe/Budapest'),
    "IS": ('IS', 'Iceland', 'Atlantic/Reykjavik'),
    "IE_SEM": ('10Y1001A1001A59C', 'Ireland (SEM) BZ / MBA', 'Europe/Dublin'),
    "IE": ('10YIE-1001A00010', 'Ireland, EirGrid CA', 'Europe/Dublin'),
    "NIE": ('10Y1001A1001A016', 'Northern Ireland, SONI CA', 'Europe/London'),
    "IT": ('10YIT-GRTN-----B', 'Italy, IT CA / MBA', 'Europe/Rome'),
    "NL": ('10YNL----------L', 'Netherlands, TenneT NL BZ / CA/ MBA', 'Europe/Amsterdam'),
    "NO_1": ('10YNO-1--------2', 'NO1 BZ / MBA', 'Europe/Oslo'),
    "NO_2": ('10YNO-2--------T', 'NO2 BZ / MBA', 'Europe/Oslo'),
    "NO_3": ('10YNO-3--------J', 'NO3 BZ / MBA', 'Europe/Oslo'),
    "NO_4": ('10YNO-4--------9', 'NO4 BZ / MBA', 'Europe/Oslo'),
    "NO_5": ('10Y1001A1001A48H', 'NO5 BZ / MBA', 'Europe/Oslo'),
    "NO": ('10YNO-0--------C', 'Norway, Norway MBA, Stattnet CA', 'Europe/Oslo'),
    "PL": ('10YPL-AREA-----S', 'Poland, PSE SA BZ / BZA / CA / MBA', 'Europe/Warsaw'),
    "PT": ('10YPT-REN------W', 'Portugal, REN BZ / CA / MBA', 'Europe/Lisbon'),
    "RO": ('10YRO-TEL------P', 'Romania, Transelectrica BZ / CA/ MBA', 'Europe/Bucharest'),
    "SE_1": ('10Y1001A1001A44P', 'SE1 BZ / MBA', 'Europe/Stockholm'),
    "SE_2": ('10Y1001A1001A45N', 'SE2 BZ / MBA', 'Europe/Stockholm'),
    "SE_3": ('10Y1001A1001A46L', 'SE3 BZ / MBA', 'Europe/Stockholm'),
    "SE_4": ('10Y1001A1001A47J', 'SE4 BZ / MBA', 'Europe/Stockholm'),
    "SK": ('10YSK-SEPS-----K', 'Slovakia, SEPS BZ / CA / MBA', 'Europe/Bratislava'),
    "SI": ('10YSI-ELES-----O', 'Slovenia, ELES BZ / CA / MBA', 'Europe/Ljubljana'),
    "ES": ('10YES-REE------0', 'Spain, REE BZ / CA / MBA', 'Europe/Madrid'),
    "SE": ('10YSE-1--------K', 'Sweden, Sweden MBA, SvK CA', 'Europe/Stockholm'),
    "CH": ('10YCH-SWISSGRIDZ', 'Switzerland, Swissgrid BZ / CA / MBA', 'Europe/Zurich'),
    "DE_TENNET": ('10YDE-EON------1', 'TenneT GER CA', 'Europe/Berlin'),
    "DE_TRANSNET": ('10YDE-ENBW-----N', 'TransnetBW CA', 'Europe/Berlin'),
    "TR": ('10YTR-TEIAS----W', 'Turkey BZ / CA / MBA', 'Europe/Istanbul'),
    "UA": ('10Y1001C--00003F', 'Ukraine, Ukraine BZ, MBA', 'Europe/Kiev'),
    "XK": ('10Y1001C--00100H', 'Kosovo/ XK CA / XK BZN', 'Europe/Rome'),
}

code_to_area = {
    value[0]: key
    for key, value in AREAS.items()
}

class EntsoeFileClientAdapter:
    """
    Adapter that exposes the same query interface as EntsoePandasClient
    but reads data from the ENTSO-E Transparency FTP/file bulk downloads.
    """

    def __init__(self, debug = False, target_zones = None, year = None):
        self.debug = debug
        self.client = client = EntsoeFileClient(
            'niklas.gerlach@email.uni-freiburg.de',
            pw)

        if year is not None:
            self.year = year

        self.load_dfs = self.download_load(year = self.year)
        self.gen_dfs = self.download_generation(year = self.year)

        self.commercial_flows_dayahead =None
        self.commercial_flows = None
        self.physical_flows = None

        self.download_scheduled_exchanges(target_zones = target_zones, year = self.year)
        self.download_physical_flows(target_zones = target_zones, year = self.year)

    def download_load(self,year = None):
        folder_name = 'ActualTotalLoad_6.1.A_r3'
        file_list = self.client.list_folder(folder_name)

        count = 0
        combined_dfs = []
        for file, id in file_list.items():
            count += 1
            if count > 10 and self.debug:
                break
            if not file.startswith(str(year)):
                continue

            print(file)
            df = self.client.download_single_file(folder_name, file)

            # only use data for bidding zones
            df = df[df["AreaTypeCode"].str.contains("BZN", na=False)]

            df = df[["DateTime(UTC)", "AreaCode", "TotalLoad[MW]"]]


            df["AreaCode"] = df["AreaCode"].map(code_to_area)
            combined_dfs.append(df)




        # Combine all downloaded files
        full_df = pd.concat(combined_dfs, ignore_index=True)

        # Drop rows where the AreaCode mapping failed
        full_df = full_df.dropna(subset=["AreaCode"])

        # Parse the UTC datetime column
        full_df["DateTime(UTC)"] = pd.to_datetime(full_df["DateTime(UTC)"], utc=True)

        # Split into per-zone DataFrames
        load_dict = {}
        for bz, group in full_df.groupby("AreaCode"):
            group = (
                group.drop(columns=["AreaCode"])
                .set_index("DateTime(UTC)")
                .rename(columns={"TotalLoad[MW]": "Actual Load"})
                .sort_index()
            )
            load_dict[f"{bz}_raw_load"] = group

        return load_dict

    def query_load(self, country_code, start=None, end = None):
        df = self.load_dfs.get(f"{country_code}_raw_load")
        if df is None:
            print("Missing load data for", country_code)
            return None

        if start is not None and end is not None:
            start_ts = pd.Timestamp(start, tz="UTC")
            end_ts = pd.Timestamp(end, tz="UTC")
            return df.loc[start_ts:end_ts].copy()

        else:
            return df

    def download_generation(self, year = None):
        folder_name = 'AggregatedGenerationPerType_16.1.B_C_r3'
        file_list = self.client.list_folder(folder_name)

        count = 0
        combined_dfs = []

        for file, id in file_list.items():
            count += 1
            if count > 10 and self.debug:
                break
            if not file.startswith(str(year)):
                continue


            print(file)
            df = self.client.download_single_file(folder_name, file)

            # Only use data for bidding zones
            df = df[df["AreaTypeCode"].str.contains("BZN", na=False)]

            # Keep only the columns we need
            df = df[[
                "DateTime(UTC)",
                "AreaCode",
                "ProductionType",
                "ActualGenerationOutput[MW]"
            ]]

            # Convert generation output to numeric
            df["ActualGenerationOutput[MW]"] = pd.to_numeric(
                df["ActualGenerationOutput[MW]"], errors="coerce"
            )

            # Map area codes to readable names and drop failed mappings
            df["AreaCode"] = df["AreaCode"].map(code_to_area)
            df = df.dropna(subset=["AreaCode"])

            combined_dfs.append(df)

        # Combine all downloaded files
        full_df = pd.concat(combined_dfs, ignore_index=True)

        # Parse the UTC datetime column
        full_df["DateTime(UTC)"] = pd.to_datetime(full_df["DateTime(UTC)"], utc=True)

        # Split into per-zone DataFrames and pivot each one
        gen_dict = {}

        for bz, group in full_df.groupby("AreaCode"):
            # Pivot so each ProductionType becomes a column
            wide = group.pivot_table(
                index="DateTime(UTC)",
                columns="ProductionType",
                values="ActualGenerationOutput[MW]",
                aggfunc="sum"
            )

            # Clean up
            wide.columns.name = None
            wide = wide.sort_index()

            gen_dict[f"{bz}_raw_generation"] = wide

        return gen_dict

    def query_generation(self, country_code, start=None, end = None, nett = False):
        df = self.gen_dfs.get(f"{country_code}_raw_generation")
        if df is None:
            print("Missing generation data for ", country_code)
            return None

        if start is not None and end is not None:
            start_ts = pd.Timestamp(start) if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert(
                'UTC')

            end_ts = pd.Timestamp(end) if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert('UTC')

            return df.loc[start_ts:end_ts].copy()

        else:
            return df

    def download_scheduled_exchanges(self, target_zones = None, year = None):
        folder_name = 'CommercialSchedules_12.1.F_r3 '
        file_list = self.client.list_folder(folder_name)

        count = 0

        # CHANGED: accumulators are now dicts of lists, keyed by bidding zone
        dfs_total = defaultdict(list)
        dfs_dayahead = defaultdict(list)

        #helper to remove columns without any real flow
        def drop_empty_flow_columns(df):
            """Drop columns that are all-NaN or all-zero."""
            if df.empty:
                return df
            df = df.dropna(axis=1, how='all')
            df = df.loc[:, ~(df == 0).all()]
            return df

        for file, id in file_list.items():
            count += 1
            if count > 10 and self.debug:
                break
            if not file.startswith(str(year)):
                continue


            print(file)
            df = self.client.download_single_file(folder_name, file)

            # Only use data for bidding zones
            df = df[df["InAreaTypeCode"].str.contains("BZN", na=False) & df["OutAreaTypeCode"].str.contains("BZN",
                                                                                                            na=False)]

            df["InAreaCode"] = df["InAreaCode"].map(code_to_area)
            df["OutAreaCode"] = df["OutAreaCode"].map(code_to_area)

            # ADDED: drop rows that could not be mapped to a known area
            df = df.dropna(subset=["InAreaCode", "OutAreaCode"])

            # ADDED: parse datetime to a proper DatetimeIndex
            df["DateTime(UTC)"] = pd.to_datetime(df["DateTime(UTC)"], utc=True)

            # ADDED: helper that turns the long-format file into a wide DataFrame
            #        with one column per flow, e.g. 'BE_NL', 'NL_BE', ...
            def pivot_flows(df, value_col):
                sub = df[["DateTime(UTC)", "OutAreaCode", "InAreaCode", value_col]].copy()
                sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
                sub = sub.dropna(subset=[value_col])
                sub["flow_col"] = sub["OutAreaCode"] + "_" + sub["InAreaCode"]
                pivoted = sub.pivot_table(
                    index="DateTime(UTC)",
                    columns="flow_col",
                    values=value_col,
                    aggfunc="first"
                )
                # CHANGED: discard columns that have no actual flow
                return drop_empty_flow_columns(pivoted)

            # CHANGED: build wide-format DataFrames instead of long-format ones
            df_total = pivot_flows(df, "TotalCapacity[MW]")
            df_dayahead = pivot_flows(df, "DayAheadCapacity[MW]")

            # ADDED: distribute the relevant flow columns to each bidding zone
            for bz in target_zones:
                cols_total = [c for c in df_total.columns
                              if c.startswith(f"{bz}_") or c.endswith(f"_{bz}")]
                if cols_total:
                    dfs_total[bz].append(df_total[cols_total])

                cols_dayahead = [c for c in df_dayahead.columns
                                 if c.startswith(f"{bz}_") or c.endswith(f"_{bz}")]
                if cols_dayahead:
                    dfs_dayahead[bz].append(df_dayahead[cols_dayahead])

        # CHANGED: final result is now a dict of per-zone DataFrames
        commercial_flows = {}
        for bz, frames in dfs_total.items():
            if frames:
                df = pd.concat(frames, axis=0)
                df = df.groupby(level=0).last().sort_index()  # resolve overlapping timestamps
                df = drop_empty_flow_columns(df)  # CHANGED: final cleanup of empty columns
            else:
                df = pd.DataFrame()
            commercial_flows[f"{bz}_raw_commercial_flows"] = df
        self.commercial_flows = commercial_flows

        commercial_flows_dayahead = {}
        for bz, frames in dfs_dayahead.items():
            if frames:
                df = pd.concat(frames, axis=0)
                df = df.groupby(level=0).last().sort_index()
                df = drop_empty_flow_columns(df)  # CHANGED: final cleanup of empty columns
            else:
                df = pd.DataFrame()
            commercial_flows_dayahead[f"{bz}_raw_commercial_flows_dayahead"] = df
        self.commercial_flows_dayahead = commercial_flows_dayahead

    def download_physical_flows(self, target_zones=None, year = None):
        folder_name = 'PhysicalFlows_12.1.G_r3 '
        file_list = self.client.list_folder(folder_name)

        count = 0

        # CHANGED: only one accumulator is needed for physical flows
        dfs_physical = defaultdict(list)

        # helper to remove columns without any real flow
        def drop_empty_flow_columns(df):
            """Drop columns that are all-NaN or all-zero."""
            if df.empty:
                return df
            df = df.dropna(axis=1, how='all')
            df = df.loc[:, ~(df == 0).all()]
            return df

        for file, id in file_list.items():
            count += 1
            if count > 10 and self.debug:
                break

            if not file.startswith(str(year)):
                continue

            print(file)
            df = self.client.download_single_file(folder_name, file)

            # Only use data for bidding zones
            df = df[df["InAreaTypeCode"].str.contains("BZN", na=False) &
                    df["OutAreaTypeCode"].str.contains("BZN", na=False)].copy()

            df["InAreaCode"] = df["InAreaCode"].map(code_to_area)
            df["OutAreaCode"] = df["OutAreaCode"].map(code_to_area)

            # drop rows that could not be mapped to a known area
            df = df.dropna(subset=["InAreaCode", "OutAreaCode"])

            # parse datetime to a proper DatetimeIndex
            df["DateTime(UTC)"] = pd.to_datetime(df["DateTime(UTC)"], utc=True)

            # helper that turns the long-format file into a wide DataFrame
            #        with one column per flow, e.g. 'BE_NL', 'NL_BE', ...
            def pivot_flows(df, value_col):
                sub = df[["DateTime(UTC)", "OutAreaCode", "InAreaCode", value_col]].copy()
                sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
                sub = sub.dropna(subset=[value_col])
                sub["flow_col"] = sub["OutAreaCode"] + "_" + sub["InAreaCode"]
                pivoted = sub.pivot_table(
                    index="DateTime(UTC)",
                    columns="flow_col",
                    values=value_col,
                    aggfunc="first"
                )
                # discard columns that have no actual flow
                return drop_empty_flow_columns(pivoted)

            # CHANGED: physical flows have only one value column, "Flow[MW]"
            df_physical = pivot_flows(df, "Flow[MW]")

            # distribute the relevant flow columns to each bidding zone
            for bz in target_zones:
                cols_physical = [c for c in df_physical.columns
                                 if c.startswith(f"{bz}_") or c.endswith(f"_{bz}")]
                if cols_physical:
                    dfs_physical[bz].append(df_physical[cols_physical])

        # final result is a dict of per-zone DataFrames
        physical_flows = {}
        for bz, frames in dfs_physical.items():
            if frames:
                df = pd.concat(frames, axis=0)
                df = df.groupby(level=0).last().sort_index()  # resolve overlapping timestamps
                df = drop_empty_flow_columns(df)  # final cleanup of empty columns
            else:
                df = pd.DataFrame()
            physical_flows[f"{bz}_raw_physical_flows"] = df

        self.physical_flows = physical_flows
