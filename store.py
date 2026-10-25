import atexit
import hashlib
import hmac
import os
import threading
import uuid

import psycopg
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

load_dotenv()

SCHEMA = "researchpilot"
PBKDF2_ITERATIONS = 200_000

_pool = None
_pool_lock = threading.Lock()


def dsn() -> str:
    url = os.environ["DATABASE_URL"]
    return url.replace("postgresql+psycopg://", "postgresql://")


def _connect():
    return psycopg.connect(dsn())


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt, PBKDF2_ITERATIONS
    )
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split("$", 1)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(digest.hex(), digest_hex)


def _set_search_path(conn):
    conn.execute(f"set search_path to {SCHEMA}, public")


def _close_pool():
    global _pool

    if _pool is not None:
        _pool.close()
        _pool = None


def build_checkpointer() -> PostgresSaver:
    global _pool

    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(
                    conninfo=dsn(),
                    max_size=5,
                    configure=_set_search_path,
                    kwargs={
                        "autocommit": True,
                        "prepare_threshold": 0,
                        "row_factory": dict_row,
                    },
                    open=True,
                )
                atexit.register(_close_pool)

    return PostgresSaver(_pool)


def init_db() -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"create schema if not exists {SCHEMA}")
            cur.execute(
                f"""
                create table if not exists {SCHEMA}.users (
                    id uuid primary key,
                    email text unique not null,
                    password_hash text not null,
                    created_at timestamptz not null default now()
                )
                """
            )
            cur.execute(
                f"""
                create table if not exists {SCHEMA}.user_threads (
                    thread_id text primary key,
                    user_id uuid not null
                        references {SCHEMA}.users(id) on delete cascade,
                    title text not null,
                    created_at timestamptz not null default now()
                )
                """
            )
            cur.execute(
                f"""
                create table if not exists {SCHEMA}.user_prefs (
                    user_id uuid not null
                        references {SCHEMA}.users(id) on delete cascade,
                    key text not null,
                    value text not null,
                    primary key (user_id, key)
                )
                """
            )
        conn.commit()

    build_checkpointer().setup()


def create_user(email: str, password: str) -> str:
    user_id = str(uuid.uuid4())

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"insert into {SCHEMA}.users (id, email, password_hash) "
                "values (%s, %s, %s)",
                (user_id, email.lower().strip(), hash_password(password)),
            )
        conn.commit()

    return user_id


def get_user(email: str):
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"select id, password_hash from {SCHEMA}.users where email = %s",
                (email.lower().strip(),),
            )
            return cur.fetchone()


def create_thread(user_id: str, title: str) -> str:
    thread_id = str(uuid.uuid4())

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"insert into {SCHEMA}.user_threads (thread_id, user_id, title) "
                "values (%s, %s, %s)",
                (thread_id, user_id, title),
            )
        conn.commit()

    return thread_id


def list_threads(user_id: str):
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"select thread_id, title, created_at from {SCHEMA}.user_threads "
                "where user_id = %s order by created_at desc",
                (user_id,),
            )
            return cur.fetchall()


def get_pref(user_id: str, key: str, default=None):
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"select value from {SCHEMA}.user_prefs "
                "where user_id = %s and key = %s",
                (user_id, key),
            )
            row = cur.fetchone()
            return row[0] if row else default


def set_pref(user_id: str, key: str, value: str) -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"insert into {SCHEMA}.user_prefs (user_id, key, value) "
                "values (%s, %s, %s) "
                "on conflict (user_id, key) do update set value = excluded.value",
                (user_id, key, value),
            )
        conn.commit()
