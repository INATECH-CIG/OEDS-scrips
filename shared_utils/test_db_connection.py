from shared_utils.postgres_utils import get_connection
from prefect import flow


@flow
def main():
    conn = get_connection()
    cur = conn.cursor()
    print('xyz')