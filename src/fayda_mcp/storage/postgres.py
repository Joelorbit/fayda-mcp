"""PostgreSQL / Neon result repository.

The repository deliberately owns no migration lifecycle. Applications should
call :meth:`create_schema` from deployment setup before serving traffic.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fayda_mcp.schemas import VerificationResult

try:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession
except ImportError:  # pragma: no cover - exercised by optional-extra tests
    AsyncSession = None  # type: ignore[assignment]
    text = None  # type: ignore[assignment]


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TERMINAL_STATUSES = ("verified", "rejected", "failed", "expired", "cancelled")


class PostgresResultRepository:
    """Durable, multi-process implementation of ``ResultRepository``."""

    def __init__(self, session_factory: Any, table_prefix: str = "fayda_") -> None:
        if AsyncSession is None or text is None:
            raise ImportError(
                "Postgres extra is not installed. Install with: pip install 'fayda-mcp[postgres]'"
            )
        if not _IDENTIFIER.fullmatch(table_prefix):
            raise ValueError("table_prefix must contain only letters, numbers, and underscores")
        if session_factory is None:
            raise ValueError("session_factory is required")
        self.session_factory = session_factory
        self.table_prefix = table_prefix
        self.requests_table = f"{table_prefix}requests"
        self.results_table = f"{table_prefix}results"

    async def create_schema(self) -> None:
        """Create tables and indexes explicitly; construction never mutates a database."""
        requests = self.requests_table
        results = self.results_table
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {requests} (
                        request_id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        principal_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        status TEXT NOT NULL,
                        expires_at TIMESTAMPTZ NOT NULL,
                        data JSONB NOT NULL,
                        UNIQUE (tenant_id, principal_id, idempotency_key)
                    )
                """))
                await session.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {results} (
                        request_id TEXT PRIMARY KEY REFERENCES {requests}(request_id) ON DELETE CASCADE,
                        expires_at TIMESTAMPTZ NOT NULL,
                        data JSONB NOT NULL
                    )
                """))
                await session.execute(text(
                    f"CREATE INDEX IF NOT EXISTS {requests}_expiry_idx ON {requests} (expires_at)"
                ))
                await session.execute(text(
                    f"CREATE INDEX IF NOT EXISTS {results}_expiry_idx ON {results} (expires_at)"
                ))

    @staticmethod
    def _expiry(value: Any, ttl_seconds: int) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str) and value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            parsed = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _json(value: Any) -> str:
        if hasattr(value, "model_dump_json"):
            return value.model_dump_json()
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")

        def _default(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        return json.dumps(value, default=_default, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _parse_json(value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)
        return value

    async def save_request(self, request_id: str, data: Dict[str, Any], ttl_seconds: int = 900) -> None:
        params = {
            "request_id": request_id,
            "tenant_id": str(data.get("tenant_id", "")),
            "principal_id": str(data.get("principal_id", "")),
            "idempotency_key": str(data.get("idempotency_key", "")),
            "status": str(data.get("status", "pending")),
            "expires_at": self._expiry(data.get("expires_at"), ttl_seconds),
            "data": self._json(data),
        }
        async with self.session_factory() as session:
            async with session.begin():
                # Reclaim expired idempotency slots before inserting. This
                # keeps the unique constraint useful without replacing a live
                # request or violating the result foreign key.
                await session.execute(text(
                    f"DELETE FROM {self.requests_table} WHERE "
                    "(request_id = :request_id OR "
                    "(tenant_id = :tenant_id AND principal_id = :principal_id "
                    "AND idempotency_key = :idempotency_key)) "
                    "AND expires_at <= CURRENT_TIMESTAMP"
                ), params)
                await session.execute(text(f"""
                    INSERT INTO {self.requests_table}
                        (request_id, tenant_id, principal_id, idempotency_key, status, expires_at, data)
                    VALUES (:request_id, :tenant_id, :principal_id, :idempotency_key,
                            :status, :expires_at, CAST(:data AS JSONB))
                    ON CONFLICT DO NOTHING
                """), params)

    async def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        async with self.session_factory() as session:
            row = (await session.execute(text(
                f"SELECT data FROM {self.requests_table} "
                "WHERE request_id = :request_id AND expires_at > CURRENT_TIMESTAMP"
            ), {"request_id": request_id})).mappings().first()
            if not row:
                return None
            data = self._parse_json(row["data"])
            return dict(data)

    async def find_by_idempotency_key(self, tenant_id: str, principal_id: str, idempotency_key: str) -> Optional[Dict[str, Any]]:
        async with self.session_factory() as session:
            row = (await session.execute(text(
                f"SELECT data FROM {self.requests_table} WHERE tenant_id = :tenant_id "
                "AND principal_id = :principal_id AND idempotency_key = :idempotency_key "
                "AND expires_at > CURRENT_TIMESTAMP"
            ), {"tenant_id": str(tenant_id), "principal_id": str(principal_id), "idempotency_key": str(idempotency_key)})).mappings().first()
            if not row:
                return None
            data = self._parse_json(row["data"])
            return dict(data)

    async def update_status(self, request_id: str, status: str) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(text(
                    f"UPDATE {self.requests_table} SET status = :status, "
                    "data = jsonb_set(data, '{status}', to_jsonb(CAST(:status AS text))) "
                    "WHERE request_id = :request_id AND expires_at > CURRENT_TIMESTAMP"
                ), {"request_id": request_id, "status": status})

    async def finalize_result(self, request_id: str, result: VerificationResult, ttl_seconds: int = 900) -> bool:
        """Finalize exactly once using a row lock inside one transaction."""
        async with self.session_factory() as session:
            async with session.begin():
                row = (await session.execute(text(
                    f"SELECT status FROM {self.requests_table} WHERE request_id = :request_id "
                    "AND expires_at > CURRENT_TIMESTAMP FOR UPDATE"
                ), {"request_id": request_id})).mappings().first()
                if not row or row["status"] in _TERMINAL_STATUSES:
                    return False
                await session.execute(text(
                    f"UPDATE {self.requests_table} SET status = :status, "
                    "data = jsonb_set(data, '{status}', to_jsonb(CAST(:status AS text))) "
                    "WHERE request_id = :request_id"
                ), {"request_id": request_id, "status": result.status})
                await session.execute(text(f"""
                    INSERT INTO {self.results_table} (request_id, expires_at, data)
                    VALUES (:request_id, :expires_at, CAST(:data AS JSONB))
                    ON CONFLICT (request_id) DO NOTHING
                """), {"request_id": request_id, "expires_at": self._expiry(result.expires_at, ttl_seconds), "data": self._json(result)})
                return True

    async def save_result(self, request_id: str, result: VerificationResult, ttl_seconds: int = 900) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(text(f"""
                    INSERT INTO {self.results_table} (request_id, expires_at, data)
                    VALUES (:request_id, :expires_at, CAST(:data AS JSONB))
                    ON CONFLICT (request_id) DO UPDATE SET expires_at = EXCLUDED.expires_at, data = EXCLUDED.data
                """), {"request_id": request_id, "expires_at": self._expiry(result.expires_at, ttl_seconds), "data": self._json(result)})
                await session.execute(text(
                    f"UPDATE {self.requests_table} SET status = :status, "
                    "data = jsonb_set(data, '{status}', to_jsonb(CAST(:status AS text))) "
                    "WHERE request_id = :request_id"
                ), {"request_id": request_id, "status": result.status})

    async def get_result(self, request_id: str) -> Optional[VerificationResult]:
        async with self.session_factory() as session:
            row = (await session.execute(text(
                f"SELECT data FROM {self.results_table} WHERE request_id = :request_id "
                "AND expires_at > CURRENT_TIMESTAMP"
            ), {"request_id": request_id})).mappings().first()
            if not row:
                return None
            data = self._parse_json(row["data"])
            return VerificationResult.model_validate(data)
