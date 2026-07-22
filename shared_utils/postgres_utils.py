from io import StringIO
import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import psycopg2
from psycopg2 import sql
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
