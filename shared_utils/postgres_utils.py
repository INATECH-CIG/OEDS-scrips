from io import StringIO
import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import time
import logging


logger = logging.getLogger(__name__)

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

def get_connection(retries: int = 5):
    conn_params = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": "open-data-17",
        "port":  "5432"
    }
    for trial in range(retries):
        try:
            conn = psycopg2.connect(**conn_params)
            return conn
        except psycopg2.OperationalError as e:
            print('error')
            logger.info(f"Timescale Connection attempt {trial} failed: {e}")

            logger.info(e)

            logger.info(conn_params)

            if trial < retries-1:
                print(f"Retrying in 3 seconds")
                time.sleep(3)
            else:
                raise Exception(f"Could not connect to the database after {retries} attempts.") from e

def ensure_schema(schemaname, readonly_user,  cur, conn):
    cur.execute(
        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}")
        .format(sql.Identifier(schemaname))
    )
    conn.commit()

    cur.execute(
        sql.SQL("GRANT USAGE ON SCHEMA {} TO {}")
        .format(
            sql.Identifier(schemaname),
            sql.Identifier(readonly_user)
        )
    )

    cur.execute(
        sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA {} GRANT SELECT ON TABLES TO {}")
        .format(
            sql.Identifier(schemaname),
            sql.Identifier(readonly_user)
        )
    )


def ensure_table(tablename, schemaname, df, cur, conn):
    """
    Ensures that for a given df, there exists a corresponding table in the timescale db.
    If the table exists, it dynamically adds any missing columns to prevent schema misalignment.
    """
    # Check if table already exists and get its current columns
    cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = %s
                """, (schemaname, tablename))
    existing_cols = {row[0].lower() for row in cur.fetchall()}

    if not existing_cols:
        # Table doesn't exist — create it from scratch
        col_defs = []
        for col, dtype in zip(df.columns, df.dtypes):
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = "BIGINT"
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = "DOUBLE PRECISION"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                sql_type = "TIMESTAMPTZ"
            else:
                sql_type = "TEXT"
            col_defs.append(f'"{col}" {sql_type}')

        create_sql = sql.SQL(
            "CREATE TABLE {}.{}({})"
        ).format(
            sql.Identifier(schemaname),
            sql.Identifier(tablename),
            sql.SQL(', ').join(sql.SQL(col_def) for col_def in col_defs)
        )
        cur.execute(create_sql)
        conn.commit()

        full_table = f'{schemaname}."{tablename}"'
        hypertable_sql = "SELECT create_hypertable(%s, 'time', if_not_exists => TRUE);"
        cur.execute(hypertable_sql, (full_table,))
        conn.commit()
    else:
        # Table exists — add any missing columns to accommodate the new DataFrame
        for col, dtype in zip(df.columns, df.dtypes):
            if col.lower() not in existing_cols:
                if pd.api.types.is_integer_dtype(dtype):
                    sql_type = "BIGINT"
                elif pd.api.types.is_float_dtype(dtype):
                    sql_type = "DOUBLE PRECISION"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    sql_type = "TIMESTAMPTZ"
                else:
                    sql_type = "TEXT"

                alter_sql = sql.SQL('ALTER TABLE {}.{} ADD COLUMN "{}" {}').format(
                    sql.Identifier(schemaname),
                    sql.Identifier(tablename),
                    sql.Identifier(col),
                    sql.SQL(sql_type)
                )
                cur.execute(alter_sql)
        conn.commit()
def df_to_timescale(df, tablename, schema_name ='public', fillna = False):
    """
    Writes a dataframe into a timescale db table
    """
    conn = get_connection()
    cur = conn.cursor()

    ensure_schema(schema_name, 'readonly', cur, conn)
    ensure_table(tablename, schema_name, df, cur, conn)

    if fillna:
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)

    ### delete old entries
    if "time" in df.columns:
        min_time = df.time.min().strftime('%Y-%m-%d %H:%M:%S%z')
        max_time = df.time.max().strftime('%Y-%m-%d %H:%M:%S%z')

        query = sql.SQL("""
                        DELETE
                        FROM {schema}.{table}
                        WHERE time >= %s
                          AND time <= %s
                        """).format(
            schema=sql.Identifier(schema_name),
            table=sql.Identifier(tablename)
        )

        cur.execute(query, (min_time, max_time))

    else:
        query = sql.SQL("""
                        DELETE
                        FROM {schema}.{table}
                        """).format(
            schema=sql.Identifier(schema_name),
            table=sql.Identifier(tablename)
        )

        cur.execute(query)
    ### end of delete

    ### insert new entries
    col_identifiers = sql.SQL(', ').join(sql.Identifier(c) for c in df.columns)

    cur.copy_expert(
        sql.SQL("COPY {}.{} ({}) FROM STDIN WITH (FORMAT CSV)")
        .format(
            sql.Identifier(schema_name),
            sql.Identifier(tablename),
            col_identifiers
        ),
        buffer
    )

    conn.commit()
    cur.close()

def ensure_flow_tracing_tables(schemaname, cur, conn):
    """
    Creates the tables for flow-tracing matrices (idempotent).
    Deliberately NOT using ensure_table(), since it does not support
    BYTEA columns or primary keys.
    NOTE: The schema name may contain hyphens (e.g. 'entsoe-flow-matrices'),
    so all identifiers are double-quoted.
    """
    full_table = f'"{schemaname}".flow_tracing_matrices'

    # Main table: one row per timestamp, 6 BYTEA columns (3 matrices x 2 methods)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS "{schemaname}".flow_tracing_matrices (
            time         TIMESTAMPTZ NOT NULL,
            q_agg        BYTEA,
            a_agg        BYTEA,
            a_inv_agg    BYTEA,
            q_dir        BYTEA,
            a_dir        BYTEA,
            a_inv_dir    BYTEA,
            n_rows       INT NOT NULL,
            n_cols       INT NOT NULL,
            patched_agg  BOOLEAN DEFAULT FALSE,
            patched_dir  BOOLEAN DEFAULT FALSE,
            PRIMARY KEY (time)
        );
    """)
    conn.commit()

    # Hypertable: table name passed as parameter (regclass), safely quoted
    cur.execute(
        "SELECT create_hypertable(%s, 'time', "
        "chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);",
        (full_table,)
    )
    conn.commit()

    # Zone index: maps matrix index <-> zone name (config.zones ordering)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS "{schemaname}".flow_tracing_zone_index (
            zone_idx  INT NOT NULL,
            zone_name TEXT NOT NULL,
            PRIMARY KEY (zone_idx)
        );
    """)
    conn.commit()

def upsert_flow_tracing_rows(rows, method, schemaname, conn):
    """
    Writes flow-tracing matrices into flow_tracing_matrices via upsert.

    rows: list of tuples (time, q_bytes, a_bytes, a_inv_bytes, n_rows, n_cols, patched)
    method: 'aggregated' or 'direct' -> determines the column suffixes.

    Uses upsert instead of delete-then-insert, since both methods push
    independently and must only overwrite their own columns. Timestamps that
    one method could not compute (LinAlgError) remain NULL in its columns.
    """
    if method == "aggregated":
        cols = "q_agg, a_agg, a_inv_agg, n_rows, patched_agg"
        updates = ("q_agg = EXCLUDED.q_agg, a_agg = EXCLUDED.a_agg, "
                   "a_inv_agg = EXCLUDED.a_inv_agg, patched_agg = EXCLUDED.patched_agg")
    elif method == "direct":
        cols = "q_dir, a_dir, a_inv_dir, n_rows, patched_dir"
        updates = ("q_dir = EXCLUDED.q_dir, a_dir = EXCLUDED.a_dir, "
                   "a_inv_dir = EXCLUDED.a_inv_dir, patched_dir = EXCLUDED.patched_dir")
    else:
        raise ValueError(f"Unknown method for flow tracing push: {method}")

    insert_sql = sql.SQL("""
        INSERT INTO {}.{} (time, {})
        VALUES %s
        ON CONFLICT (time) DO UPDATE SET
            {},
            n_rows = EXCLUDED.n_rows,
            n_cols = EXCLUDED.n_cols;
    """).format(
        sql.Identifier(schemaname),
        sql.Identifier("flow_tracing_matrices"),
        sql.SQL(cols),
        sql.SQL(updates)
    )

    data = [
        (
            t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
            psycopg2.Binary(q_b), psycopg2.Binary(a_b), psycopg2.Binary(a_inv_b),
            n_rows, n_cols, patched
        )
        for (t, q_b, a_b, a_inv_b, n_rows, n_cols, patched) in rows
    ]

    cur = conn.cursor()
    execute_values(cur, insert_sql.as_string(conn), data, page_size=500)
    conn.commit()
    cur.close()