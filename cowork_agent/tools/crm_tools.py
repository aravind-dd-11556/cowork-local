"""
CRM Integration Tools — Sprint 36.

Mirrors real Cowork's CRM MCP connectors (e.g. ZohoCRM).
Provides a generic CRM bridge with pluggable backends:
  1. CrmGetRecordsTool     — Retrieve records from a CRM module
  2. CrmSearchRecordsTool  — Search records by criteria/email/phone/word
  3. CrmCreateRecordsTool  — Create new records in a CRM module
  4. CrmUpdateRecordsTool  — Update existing records
  5. CrmDeleteRecordsTool  — Delete records by ID
  6. CrmUpsertRecordsTool  — Insert or update based on duplicate check

The tools accept a CRM backend (callable/adapter) for real integrations
and fall back to simulated in-memory storage for testing.
"""

from __future__ import annotations
import copy
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import BaseTool

logger = logging.getLogger(__name__)


# ── In-memory CRM store (simulated backend) ────────────────────────

@dataclass
class CrmRecord:
    """Single CRM record."""
    id: str
    module: str
    data: Dict[str, Any] = field(default_factory=dict)
    created_time: float = 0.0
    modified_time: float = 0.0
    deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        out = {"id": self.id, "module": self.module, **self.data}
        out["Created_Time"] = self.created_time
        out["Modified_Time"] = self.modified_time
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any], module: str = "") -> "CrmRecord":
        """Deserialize from dict."""
        rec_id = d.get("id", str(uuid.uuid4())[:18])
        mod = d.get("module", module)
        data = {k: v for k, v in d.items() if k not in ("id", "module", "Created_Time", "Modified_Time")}
        return cls(
            id=rec_id, module=mod, data=data,
            created_time=d.get("Created_Time", time.time()),
            modified_time=d.get("Modified_Time", time.time()),
        )


class InMemoryCrmBackend:
    """In-memory CRM backend for testing and simulation."""

    def __init__(self):
        self._records: Dict[str, Dict[str, CrmRecord]] = {}  # module -> {id -> record}

    def get_records(self, module: str, *, ids: Optional[List[str]] = None,
                    fields: Optional[List[str]] = None,
                    page: int = 1, per_page: int = 200,
                    sort_by: str = "id", sort_order: str = "desc") -> List[Dict]:
        """Get records from a module."""
        mod_records = self._records.get(module, {})
        records = [r for r in mod_records.values() if not r.deleted]

        if ids:
            records = [r for r in records if r.id in ids]

        # Sort
        reverse = sort_order == "desc"
        if sort_by == "Created_Time":
            records.sort(key=lambda r: r.created_time, reverse=reverse)
        elif sort_by == "Modified_Time":
            records.sort(key=lambda r: r.modified_time, reverse=reverse)
        else:
            records.sort(key=lambda r: r.id, reverse=reverse)

        # Paginate
        start = (page - 1) * per_page
        page_records = records[start:start + per_page]

        result = []
        for r in page_records:
            d = r.to_dict()
            if fields:
                d = {k: v for k, v in d.items() if k in fields or k == "id"}
            result.append(d)
        return result

    def search_records(self, module: str, *,
                       criteria: Optional[str] = None,
                       email: Optional[str] = None,
                       phone: Optional[str] = None,
                       word: Optional[str] = None) -> List[Dict]:
        """Search records by criteria."""
        mod_records = self._records.get(module, {})
        records = [r for r in mod_records.values() if not r.deleted]
        matches = []

        for r in records:
            if word and word.lower() in str(r.data).lower():
                matches.append(r.to_dict())
            elif email:
                for v in r.data.values():
                    if isinstance(v, str) and email.lower() in v.lower():
                        matches.append(r.to_dict())
                        break
            elif phone:
                for v in r.data.values():
                    if isinstance(v, str) and phone in v:
                        matches.append(r.to_dict())
                        break
            elif criteria:
                # Simple criteria: field:equals:value
                matches.append(r.to_dict())  # Simplified: return all

        return matches

    def create_records(self, module: str, data_list: List[Dict]) -> List[Dict]:
        """Create records in a module."""
        if module not in self._records:
            self._records[module] = {}

        results = []
        for data in data_list:
            rec_id = data.get("id", str(uuid.uuid4())[:18])
            now = time.time()
            record = CrmRecord(
                id=rec_id, module=module,
                data={k: v for k, v in data.items() if k != "id"},
                created_time=now, modified_time=now,
            )
            self._records[module][rec_id] = record
            results.append({
                "status": "success",
                "code": "SUCCESS",
                "details": {"id": rec_id},
            })
        return results

    def update_records(self, module: str, data_list: List[Dict]) -> List[Dict]:
        """Update existing records."""
        mod_records = self._records.get(module, {})
        results = []
        for data in data_list:
            rec_id = data.get("id")
            if not rec_id or rec_id not in mod_records:
                results.append({
                    "status": "error",
                    "code": "INVALID_DATA",
                    "details": {"id": rec_id},
                    "message": f"Record {rec_id} not found.",
                })
                continue
            record = mod_records[rec_id]
            for k, v in data.items():
                if k != "id":
                    record.data[k] = v
            record.modified_time = time.time()
            results.append({
                "status": "success",
                "code": "SUCCESS",
                "details": {"id": rec_id},
            })
        return results

    def delete_records(self, module: str, ids: List[str]) -> List[Dict]:
        """Delete records by ID."""
        mod_records = self._records.get(module, {})
        results = []
        for rec_id in ids:
            if rec_id in mod_records and not mod_records[rec_id].deleted:
                mod_records[rec_id].deleted = True
                results.append({
                    "status": "success",
                    "code": "SUCCESS",
                    "details": {"id": rec_id},
                })
            else:
                results.append({
                    "status": "error",
                    "code": "INVALID_DATA",
                    "details": {"id": rec_id},
                    "message": f"Record {rec_id} not found.",
                })
        return results

    def upsert_records(self, module: str, data_list: List[Dict],
                       duplicate_check_fields: Optional[List[str]] = None) -> List[Dict]:
        """Upsert records (create or update based on duplicate check)."""
        if module not in self._records:
            self._records[module] = {}

        results = []
        for data in data_list:
            existing = None
            if duplicate_check_fields:
                for rec in self._records[module].values():
                    if rec.deleted:
                        continue
                    match = all(
                        rec.data.get(f) == data.get(f)
                        for f in duplicate_check_fields
                        if f in data
                    )
                    if match:
                        existing = rec
                        break

            if existing:
                for k, v in data.items():
                    if k != "id":
                        existing.data[k] = v
                existing.modified_time = time.time()
                results.append({
                    "status": "success",
                    "code": "SUCCESS",
                    "action": "update",
                    "details": {"id": existing.id},
                })
            else:
                rec_id = data.get("id", str(uuid.uuid4())[:18])
                now = time.time()
                record = CrmRecord(
                    id=rec_id, module=module,
                    data={k: v for k, v in data.items() if k != "id"},
                    created_time=now, modified_time=now,
                )
                self._records[module][rec_id] = record
                results.append({
                    "status": "success",
                    "code": "SUCCESS",
                    "action": "insert",
                    "details": {"id": rec_id},
                })
        return results


# ── Standard CRM modules ───────────────────────────────────────────

STANDARD_MODULES = frozenset({
    "Leads", "Accounts", "Contacts", "Deals", "Campaigns",
    "Tasks", "Cases", "Meetings", "Calls", "Solutions",
    "Products", "Vendors", "Price_Books", "Quotes",
    "Sales_Orders", "Purchase_Orders", "Invoices",
    "Appointments", "Services",
})


# ═══════════════════════════════════════════════════════════════════════
# Tool 1: Get Records
# ═══════════════════════════════════════════════════════════════════════

class CrmGetRecordsTool(BaseTool):
    """Get records from a CRM module."""
    name = "crm_get_records"
    description = (
        "Retrieve records from a CRM module. Supports pagination, "
        "field selection, sorting, and filtering by IDs."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "module": {
                "type": "string",
                "description": "CRM module API name (e.g. Leads, Contacts, Deals).",
            },
            "ids": {
                "type": "string",
                "description": "Comma-separated record IDs to retrieve.",
            },
            "fields": {
                "type": "string",
                "description": "Comma-separated field API names to return.",
            },
            "page": {"type": "integer", "description": "Page number (default 1)."},
            "per_page": {"type": "integer", "description": "Records per page (default 200)."},
            "sort_by": {"type": "string", "description": "Sort field."},
            "sort_order": {"type": "string", "enum": ["asc", "desc"]},
        },
        "required": ["module"],
    }

    def __init__(self, crm_backend: Optional[Any] = None):
        self._backend = crm_backend or InMemoryCrmBackend()

    async def execute(self, *, progress_callback=None, module=None,
                      ids=None, fields=None, page=1, per_page=200,
                      sort_by="id", sort_order="desc", **kwargs) -> "ToolResult":
        """Retrieve CRM records."""
        if not module:
            return self._error("'module' parameter is required.")

        id_list = [i.strip() for i in ids.split(",")] if ids else None
        field_list = [f.strip() for f in fields.split(",")] if fields else None

        try:
            records = self._backend.get_records(
                module, ids=id_list, fields=field_list,
                page=int(page), per_page=int(per_page),
                sort_by=sort_by, sort_order=sort_order,
            )
            if not records:
                return self._success(
                    f"No records found in {module}.",
                    module=module, record_count=0,
                )

            lines = [f"{module} records ({len(records)}):\n"]
            for rec in records:
                rec_id = rec.get("id", "?")
                name = rec.get("Full_Name") or rec.get("Name") or rec.get("Last_Name", "")
                lines.append(f"  [{rec_id}] {name}")

            return self._success(
                "\n".join(lines),
                module=module,
                record_count=len(records),
                records=records,
            )
        except Exception as e:
            return self._error(f"CRM get records error: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Tool 2: Search Records
# ═══════════════════════════════════════════════════════════════════════

class CrmSearchRecordsTool(BaseTool):
    """Search CRM records by criteria, email, phone, or keyword."""
    name = "crm_search_records"
    description = (
        "Search records in a CRM module using criteria, email, phone, "
        "or word search. At least one search parameter is required."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "module": {"type": "string", "description": "CRM module API name."},
            "criteria": {"type": "string", "description": "Search criteria expression."},
            "email": {"type": "string", "description": "Search by email."},
            "phone": {"type": "string", "description": "Search by phone."},
            "word": {"type": "string", "description": "Keyword search."},
            "fields": {"type": "string", "description": "Comma-separated fields to return."},
            "page": {"type": "integer"},
            "per_page": {"type": "integer"},
        },
        "required": ["module"],
    }

    def __init__(self, crm_backend: Optional[Any] = None):
        self._backend = crm_backend or InMemoryCrmBackend()

    async def execute(self, *, progress_callback=None, module=None,
                      criteria=None, email=None, phone=None, word=None,
                      fields=None, page=1, per_page=200, **kwargs) -> "ToolResult":
        """Search CRM records."""
        if not module:
            return self._error("'module' parameter is required.")
        if not any([criteria, email, phone, word]):
            return self._error(
                "At least one search parameter (criteria, email, phone, word) is required."
            )

        try:
            results = self._backend.search_records(
                module, criteria=criteria, email=email, phone=phone, word=word,
            )

            if not results:
                return self._success(
                    f"No matching records found in {module}.",
                    module=module, record_count=0,
                )

            lines = [f"Search results in {module} ({len(results)}):\n"]
            for rec in results:
                rec_id = rec.get("id", "?")
                name = rec.get("Full_Name") or rec.get("Name") or rec.get("Last_Name", "")
                lines.append(f"  [{rec_id}] {name}")

            return self._success(
                "\n".join(lines),
                module=module,
                record_count=len(results),
                records=results,
            )
        except Exception as e:
            return self._error(f"CRM search error: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Tool 3: Create Records
# ═══════════════════════════════════════════════════════════════════════

class CrmCreateRecordsTool(BaseTool):
    """Create records in a CRM module."""
    name = "crm_create_records"
    description = (
        "Create one or more new records in a CRM module. "
        "Provide record data as a list of objects."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "module": {"type": "string", "description": "CRM module API name."},
            "data": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of record objects to create.",
            },
            "trigger": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Automation triggers to invoke.",
            },
        },
        "required": ["module", "data"],
    }

    def __init__(self, crm_backend: Optional[Any] = None):
        self._backend = crm_backend or InMemoryCrmBackend()

    async def execute(self, *, progress_callback=None, module=None,
                      data=None, trigger=None, **kwargs) -> "ToolResult":
        """Create CRM records."""
        if not module:
            return self._error("'module' parameter is required.")
        if not data or not isinstance(data, list):
            return self._error("'data' must be a non-empty list of records.")

        try:
            results = self._backend.create_records(module, data)
            success_count = sum(1 for r in results if r.get("status") == "success")

            return self._success(
                f"Created {success_count}/{len(data)} records in {module}.",
                module=module,
                created_count=success_count,
                total=len(data),
                results=results,
            )
        except Exception as e:
            return self._error(f"CRM create error: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Tool 4: Update Records
# ═══════════════════════════════════════════════════════════════════════

class CrmUpdateRecordsTool(BaseTool):
    """Update existing CRM records."""
    name = "crm_update_records"
    description = (
        "Update one or more existing records in a CRM module. "
        "Each record must include its ID."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "module": {"type": "string", "description": "CRM module API name."},
            "data": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of record objects with IDs to update.",
            },
            "trigger": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Automation triggers to invoke.",
            },
        },
        "required": ["module", "data"],
    }

    def __init__(self, crm_backend: Optional[Any] = None):
        self._backend = crm_backend or InMemoryCrmBackend()

    async def execute(self, *, progress_callback=None, module=None,
                      data=None, trigger=None, **kwargs) -> "ToolResult":
        """Update CRM records."""
        if not module:
            return self._error("'module' parameter is required.")
        if not data or not isinstance(data, list):
            return self._error("'data' must be a non-empty list of records.")

        # Validate all records have IDs
        for rec in data:
            if "id" not in rec:
                return self._error("Each record must include an 'id' field.")

        try:
            results = self._backend.update_records(module, data)
            success_count = sum(1 for r in results if r.get("status") == "success")

            return self._success(
                f"Updated {success_count}/{len(data)} records in {module}.",
                module=module,
                updated_count=success_count,
                total=len(data),
                results=results,
            )
        except Exception as e:
            return self._error(f"CRM update error: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Tool 5: Delete Records
# ═══════════════════════════════════════════════════════════════════════

class CrmDeleteRecordsTool(BaseTool):
    """Delete records from a CRM module."""
    name = "crm_delete_records"
    description = (
        "Permanently delete one or more records from a CRM module "
        "using comma-separated record IDs."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "module": {"type": "string", "description": "CRM module API name."},
            "ids": {"type": "string", "description": "Comma-separated record IDs to delete."},
        },
        "required": ["module", "ids"],
    }

    def __init__(self, crm_backend: Optional[Any] = None):
        self._backend = crm_backend or InMemoryCrmBackend()

    async def execute(self, *, progress_callback=None, module=None,
                      ids=None, **kwargs) -> "ToolResult":
        """Delete CRM records."""
        if not module:
            return self._error("'module' parameter is required.")
        if not ids:
            return self._error("'ids' parameter is required.")

        id_list = [i.strip() for i in ids.split(",")]

        try:
            results = self._backend.delete_records(module, id_list)
            success_count = sum(1 for r in results if r.get("status") == "success")

            return self._success(
                f"Deleted {success_count}/{len(id_list)} records from {module}.",
                module=module,
                deleted_count=success_count,
                total=len(id_list),
                results=results,
            )
        except Exception as e:
            return self._error(f"CRM delete error: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Tool 6: Upsert Records
# ═══════════════════════════════════════════════════════════════════════

class CrmUpsertRecordsTool(BaseTool):
    """Upsert records (create or update based on duplicate check)."""
    name = "crm_upsert_records"
    description = (
        "Insert new records or update existing ones based on duplicate "
        "check field values. Prevents duplicate creation."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "module": {"type": "string", "description": "CRM module API name."},
            "data": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of record objects to upsert.",
            },
            "duplicate_check_fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Fields used to detect duplicates.",
            },
            "trigger": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Automation triggers to invoke.",
            },
        },
        "required": ["module", "data"],
    }

    def __init__(self, crm_backend: Optional[Any] = None):
        self._backend = crm_backend or InMemoryCrmBackend()

    async def execute(self, *, progress_callback=None, module=None,
                      data=None, duplicate_check_fields=None,
                      trigger=None, **kwargs) -> "ToolResult":
        """Upsert CRM records."""
        if not module:
            return self._error("'module' parameter is required.")
        if not data or not isinstance(data, list):
            return self._error("'data' must be a non-empty list of records.")

        try:
            results = self._backend.upsert_records(
                module, data, duplicate_check_fields=duplicate_check_fields,
            )
            inserts = sum(1 for r in results if r.get("action") == "insert")
            updates = sum(1 for r in results if r.get("action") == "update")

            return self._success(
                f"Upserted {len(data)} records in {module}: "
                f"{inserts} inserted, {updates} updated.",
                module=module,
                insert_count=inserts,
                update_count=updates,
                total=len(data),
                results=results,
            )
        except Exception as e:
            return self._error(f"CRM upsert error: {e}")
