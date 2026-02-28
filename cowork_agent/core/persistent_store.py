"""
Persistent Store — SQLite-based storage for metrics, audit logs, and benchmarks.

Provides durable persistence with time-range queries, aggregations, and
automatic cleanup/rotation of old data.

Sprint 19 (Persistent Storage) Module 1.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# ── PersistentStore ──────────────────────────────────────────────

class PersistentStore:
    """
    SQLite-based persistent storage for observability data.

    Creates and manages a SQLite database with tables for metrics,
    audit events, and benchmark runs.

    Usage::

        store = PersistentStore("/path/to/.cowork/metrics")
        store.metrics.insert_token_usage("anthropic", "claude", 100, 50)
        rows = store.metrics.query_token_usage(provider="anthropic")
    """

    def __init__(self, base_path: str, db_name: str = "metrics.db"):
        self._base_path = base_path
        self._db_path = os.path.join(base_path, db_name)

        # Ensure directory exists
        os.makedirs(base_path, exist_ok=True)

        # Initialize DB and tables
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._initialize_schema()

        # Table managers
        self.metrics = MetricsTable(self)
        self.audit = AuditTable(self)
        self.benchmarks = BenchmarkTable(self)

    def _connect(self) -> None:
        """Create or reopen the SQLite connection."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    def _initialize_schema(self) -> None:
        """Create tables if they don't exist."""
        with self.transaction() as cur:
            # Schema version tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at REAL NOT NULL
                )
            """)

            # Check if schema already applied
            cur.execute("SELECT MAX(version) FROM schema_version")
            row = cur.fetchone()
            current_version = row[0] if row[0] is not None else 0

            if current_version < SCHEMA_VERSION:
                self._apply_schema_v1(cur)
                cur.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, time.time()),
                )

    def _apply_schema_v1(self, cur: sqlite3.Cursor) -> None:
        """Apply v1 schema: metrics, audit, benchmark tables."""
        # Token usage
        cur.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                timestamp REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_usage_ts
            ON token_usage(timestamp)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_usage_provider
            ON token_usage(provider)
        """)

        # Provider errors
        cur.execute("""
            CREATE TABLE IF NOT EXISTS provider_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                error_type TEXT NOT NULL,
                error_message TEXT,
                timestamp REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_errors_ts
            ON provider_errors(timestamp)
        """)

        # Provider calls
        cur.execute("""
            CREATE TABLE IF NOT EXISTS provider_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                success INTEGER NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_calls_ts
            ON provider_calls(timestamp)
        """)

        # Audit events
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                component TEXT NOT NULL,
                description TEXT NOT NULL,
                tool_name TEXT DEFAULT '',
                blocked INTEGER DEFAULT 0,
                correlation_id TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                timestamp REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_ts
            ON audit_events(timestamp)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_severity
            ON audit_events(severity)
        """)

        # Benchmark runs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                component TEXT DEFAULT '',
                success INTEGER NOT NULL,
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                timestamp REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_bench_name
            ON benchmark_runs(name)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_bench_ts
            ON benchmark_runs(timestamp)
        """)

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def stats(self) -> Dict[str, Any]:
        """Return storage statistics."""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM token_usage")
        token_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM provider_errors")
        error_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM provider_calls")
        call_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM audit_events")
        audit_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM benchmark_runs")
        bench_count = cur.fetchone()[0]

        db_size = os.path.getsize(self._db_path) if os.path.exists(self._db_path) else 0

        return {
            "db_path": self._db_path,
            "db_size_bytes": db_size,
            "schema_version": SCHEMA_VERSION,
            "token_usage_rows": token_count,
            "error_rows": error_count,
            "call_rows": call_count,
            "audit_event_rows": audit_count,
            "benchmark_run_rows": bench_count,
        }


# ── MetricsTable ─────────────────────────────────────────────────

class MetricsTable:
    """Manages metrics persistence (token usage, errors, provider calls)."""

    def __init__(self, store: PersistentStore):
        self._store = store

    def insert_token_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        timestamp: Optional[float] = None,
    ) -> int:
        """Insert a token usage record. Returns row ID."""
        ts = timestamp or time.time()
        with self._store.transaction() as cur:
            cur.execute(
                """INSERT INTO token_usage
                   (provider, model, input_tokens, output_tokens,
                    cache_read_tokens, cache_write_tokens, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (provider, model, input_tokens, output_tokens,
                 cache_read_tokens, cache_write_tokens, ts),
            )
            return cur.lastrowid

    def insert_error(
        self,
        provider: str,
        error_type: str,
        error_message: str = "",
        timestamp: Optional[float] = None,
    ) -> int:
        """Insert an error record."""
        ts = timestamp or time.time()
        with self._store.transaction() as cur:
            cur.execute(
                """INSERT INTO provider_errors
                   (provider, error_type, error_message, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (provider, error_type, error_message, ts),
            )
            return cur.lastrowid

    def insert_provider_call(
        self,
        provider: str,
        duration_ms: float,
        success: bool,
        timestamp: Optional[float] = None,
    ) -> int:
        """Insert a provider call record."""
        ts = timestamp or time.time()
        with self._store.transaction() as cur:
            cur.execute(
                """INSERT INTO provider_calls
                   (provider, duration_ms, success, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (provider, duration_ms, 1 if success else 0, ts),
            )
            return cur.lastrowid

    def query_token_usage(
        self,
        provider: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query token usage records with filters."""
        query = "SELECT * FROM token_usage WHERE 1=1"
        params: list = []

        if provider:
            query += " AND provider = ?"
            params.append(provider)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        if until:
            query += " AND timestamp <= ?"
            params.append(until)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cur = self._store._conn.cursor()
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    def query_errors(
        self,
        provider: Optional[str] = None,
        error_type: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query error records."""
        query = "SELECT * FROM provider_errors WHERE 1=1"
        params: list = []

        if provider:
            query += " AND provider = ?"
            params.append(provider)
        if error_type:
            query += " AND error_type = ?"
            params.append(error_type)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cur = self._store._conn.cursor()
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    def query_provider_calls(
        self,
        provider: Optional[str] = None,
        since: Optional[float] = None,
        success_only: Optional[bool] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query provider call records."""
        query = "SELECT * FROM provider_calls WHERE 1=1"
        params: list = []

        if provider:
            query += " AND provider = ?"
            params.append(provider)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        if success_only is not None:
            query += " AND success = ?"
            params.append(1 if success_only else 0)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cur = self._store._conn.cursor()
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    def aggregate_daily(
        self,
        metric: str = "token_usage",
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics by day."""
        if metric == "token_usage":
            query = """
                SELECT
                    date(timestamp, 'unixepoch') as day,
                    SUM(input_tokens) as total_input,
                    SUM(output_tokens) as total_output,
                    COUNT(*) as call_count
                FROM token_usage
                WHERE 1=1
            """
        elif metric == "errors":
            query = """
                SELECT
                    date(timestamp, 'unixepoch') as day,
                    COUNT(*) as error_count,
                    error_type
                FROM provider_errors
                WHERE 1=1
            """
        elif metric == "calls":
            query = """
                SELECT
                    date(timestamp, 'unixepoch') as day,
                    COUNT(*) as total_calls,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    AVG(duration_ms) as avg_duration_ms
                FROM provider_calls
                WHERE 1=1
            """
        else:
            return []

        params: list = []
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        if until:
            query += " AND timestamp <= ?"
            params.append(until)

        if metric == "errors":
            query += " GROUP BY day, error_type ORDER BY day"
        else:
            query += " GROUP BY day ORDER BY day"

        cur = self._store._conn.cursor()
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    def delete_before(self, timestamp: float) -> int:
        """Delete all metric records older than timestamp. Returns rows deleted."""
        total = 0
        with self._store.transaction() as cur:
            cur.execute("DELETE FROM token_usage WHERE timestamp < ?", (timestamp,))
            total += cur.rowcount
            cur.execute("DELETE FROM provider_errors WHERE timestamp < ?", (timestamp,))
            total += cur.rowcount
            cur.execute("DELETE FROM provider_calls WHERE timestamp < ?", (timestamp,))
            total += cur.rowcount
        return total


# ── AuditTable ───────────────────────────────────────────────────

class AuditTable:
    """Manages audit event persistence."""

    def __init__(self, store: PersistentStore):
        self._store = store

    def insert_event(
        self,
        event_type: str,
        severity: str,
        component: str,
        description: str,
        tool_name: str = "",
        blocked: bool = False,
        correlation_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """Insert an audit event."""
        ts = timestamp or time.time()
        meta_str = json.dumps(metadata or {})
        with self._store.transaction() as cur:
            cur.execute(
                """INSERT INTO audit_events
                   (event_type, severity, component, description,
                    tool_name, blocked, correlation_id, metadata, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_type, severity, component, description,
                 tool_name, 1 if blocked else 0, correlation_id,
                 meta_str, ts),
            )
            return cur.lastrowid

    def query_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        component: Optional[str] = None,
        since: Optional[float] = None,
        blocked_only: bool = False,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query audit events with filters."""
        query = "SELECT * FROM audit_events WHERE 1=1"
        params: list = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if component:
            query += " AND component = ?"
            params.append(component)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        if blocked_only:
            query += " AND blocked = 1"

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cur = self._store._conn.cursor()
        cur.execute(query, params)
        rows = []
        for row in cur.fetchall():
            d = dict(row)
            d["blocked"] = bool(d["blocked"])
            try:
                d["metadata"] = json.loads(d.get("metadata", "{}"))
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
            rows.append(d)
        return rows

    def count_by_severity(
        self, since: Optional[float] = None,
    ) -> Dict[str, int]:
        """Count events grouped by severity."""
        query = "SELECT severity, COUNT(*) as cnt FROM audit_events WHERE 1=1"
        params: list = []
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        query += " GROUP BY severity"

        cur = self._store._conn.cursor()
        cur.execute(query, params)
        return {row["severity"]: row["cnt"] for row in cur.fetchall()}

    def delete_before(self, timestamp: float) -> int:
        """Delete audit events older than timestamp."""
        with self._store.transaction() as cur:
            cur.execute("DELETE FROM audit_events WHERE timestamp < ?", (timestamp,))
            return cur.rowcount


# ── BenchmarkTable ───────────────────────────────────────────────

class BenchmarkTable:
    """Manages benchmark run persistence."""

    def __init__(self, store: PersistentStore):
        self._store = store

    def insert_run(
        self,
        name: str,
        duration_ms: float,
        component: str = "",
        success: bool = True,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """Insert a benchmark run."""
        ts = timestamp or time.time()
        tags_str = json.dumps(tags or [])
        meta_str = json.dumps(metadata or {})
        with self._store.transaction() as cur:
            cur.execute(
                """INSERT INTO benchmark_runs
                   (name, duration_ms, component, success, tags, metadata, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (name, duration_ms, component, 1 if success else 0,
                 tags_str, meta_str, ts),
            )
            return cur.lastrowid

    def query_runs(
        self,
        name: Optional[str] = None,
        component: Optional[str] = None,
        since: Optional[float] = None,
        success_only: Optional[bool] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query benchmark runs."""
        query = "SELECT * FROM benchmark_runs WHERE 1=1"
        params: list = []

        if name:
            query += " AND name = ?"
            params.append(name)
        if component:
            query += " AND component = ?"
            params.append(component)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        if success_only is not None:
            query += " AND success = ?"
            params.append(1 if success_only else 0)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cur = self._store._conn.cursor()
        cur.execute(query, params)
        rows = []
        for row in cur.fetchall():
            d = dict(row)
            d["success"] = bool(d["success"])
            try:
                d["tags"] = json.loads(d.get("tags", "[]"))
            except (json.JSONDecodeError, TypeError):
                d["tags"] = []
            try:
                d["metadata"] = json.loads(d.get("metadata", "{}"))
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
            rows.append(d)
        return rows

    def get_stats(self, name: str, since: Optional[float] = None) -> Dict[str, Any]:
        """Get aggregate statistics for a benchmark name."""
        query = "SELECT duration_ms, success FROM benchmark_runs WHERE name = ?"
        params: list = [name]
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        query += " ORDER BY timestamp"

        cur = self._store._conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()

        if not rows:
            return {"name": name, "count": 0}

        durations = [r["duration_ms"] for r in rows]
        successes = [bool(r["success"]) for r in rows]
        sorted_d = sorted(durations)
        n = len(sorted_d)

        return {
            "name": name,
            "count": n,
            "min_ms": min(sorted_d),
            "max_ms": max(sorted_d),
            "avg_ms": statistics.mean(sorted_d),
            "median_ms": statistics.median(sorted_d),
            "p95_ms": sorted_d[int(n * 0.95)] if n >= 2 else sorted_d[-1],
            "p99_ms": sorted_d[int(n * 0.99)] if n >= 2 else sorted_d[-1],
            "std_dev_ms": statistics.stdev(sorted_d) if n >= 2 else 0.0,
            "success_rate": sum(successes) / len(successes),
        }

    def delete_before(self, timestamp: float) -> int:
        """Delete benchmark runs older than timestamp."""
        with self._store.transaction() as cur:
            cur.execute("DELETE FROM benchmark_runs WHERE timestamp < ?", (timestamp,))
            return cur.rowcount
