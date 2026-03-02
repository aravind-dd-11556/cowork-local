"""
Sprint 36 Tests — CRM Integration Tools.

Tests for:
  - InMemoryCrmBackend
  - CrmGetRecordsTool
  - CrmSearchRecordsTool
  - CrmCreateRecordsTool
  - CrmUpdateRecordsTool
  - CrmDeleteRecordsTool
  - CrmUpsertRecordsTool
  - Main.py wiring
  - Edge cases

~85 tests total.
"""

import asyncio
import unittest

from cowork_agent.tools.crm_tools import (
    CrmCreateRecordsTool,
    CrmDeleteRecordsTool,
    CrmGetRecordsTool,
    CrmSearchRecordsTool,
    CrmUpdateRecordsTool,
    CrmUpsertRecordsTool,
    CrmRecord,
    InMemoryCrmBackend,
    STANDARD_MODULES,
)


def run(coro):
    """Run async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _populated_backend():
    """Create a backend with sample data."""
    backend = InMemoryCrmBackend()
    backend.create_records("Leads", [
        {"id": "L001", "Last_Name": "Smith", "Email": "smith@example.com", "Phone": "555-0101"},
        {"id": "L002", "Last_Name": "Jones", "Email": "jones@test.org", "Phone": "555-0102"},
        {"id": "L003", "Last_Name": "Brown", "Email": "brown@example.com", "Phone": "555-0103"},
    ])
    backend.create_records("Contacts", [
        {"id": "C001", "Last_Name": "Adams", "Email": "adams@corp.com"},
        {"id": "C002", "Last_Name": "Baker", "Email": "baker@corp.com"},
    ])
    return backend


# ═══════════════════════════════════════════════════════════════════════
# InMemoryCrmBackend Tests
# ═══════════════════════════════════════════════════════════════════════


class TestInMemoryCrmBackend(unittest.TestCase):
    """Tests for the in-memory CRM backend."""

    def test_create_and_get(self):
        """Test creating and retrieving records."""
        backend = InMemoryCrmBackend()
        backend.create_records("Leads", [{"id": "L1", "Last_Name": "Test"}])
        records = backend.get_records("Leads")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "L1")

    def test_get_empty_module(self):
        """Test getting from empty module."""
        backend = InMemoryCrmBackend()
        records = backend.get_records("Leads")
        self.assertEqual(records, [])

    def test_get_by_ids(self):
        """Test filtering by IDs."""
        backend = _populated_backend()
        records = backend.get_records("Leads", ids=["L001", "L003"])
        self.assertEqual(len(records), 2)

    def test_get_with_fields(self):
        """Test field selection."""
        backend = _populated_backend()
        records = backend.get_records("Leads", fields=["Last_Name"])
        for r in records:
            self.assertIn("id", r)
            self.assertIn("Last_Name", r)
            self.assertNotIn("Phone", r)

    def test_get_pagination(self):
        """Test pagination."""
        backend = _populated_backend()
        page1 = backend.get_records("Leads", page=1, per_page=2)
        page2 = backend.get_records("Leads", page=2, per_page=2)
        self.assertEqual(len(page1), 2)
        self.assertEqual(len(page2), 1)

    def test_get_sort_asc(self):
        """Test ascending sort by ID."""
        backend = _populated_backend()
        records = backend.get_records("Leads", sort_order="asc")
        ids = [r["id"] for r in records]
        self.assertEqual(ids, sorted(ids))

    def test_search_by_word(self):
        """Test word search."""
        backend = _populated_backend()
        results = backend.search_records("Leads", word="Smith")
        self.assertEqual(len(results), 1)

    def test_search_by_email(self):
        """Test email search."""
        backend = _populated_backend()
        results = backend.search_records("Leads", email="example.com")
        self.assertEqual(len(results), 2)  # smith and brown

    def test_search_by_phone(self):
        """Test phone search."""
        backend = _populated_backend()
        results = backend.search_records("Leads", phone="555-0102")
        self.assertEqual(len(results), 1)

    def test_update_record(self):
        """Test updating a record."""
        backend = _populated_backend()
        results = backend.update_records("Leads", [
            {"id": "L001", "Last_Name": "Smith-Updated"},
        ])
        self.assertEqual(results[0]["status"], "success")
        records = backend.get_records("Leads", ids=["L001"])
        self.assertEqual(records[0]["Last_Name"], "Smith-Updated")

    def test_update_nonexistent(self):
        """Test updating non-existent record."""
        backend = _populated_backend()
        results = backend.update_records("Leads", [
            {"id": "L999", "Last_Name": "Ghost"},
        ])
        self.assertEqual(results[0]["status"], "error")

    def test_delete_record(self):
        """Test deleting a record."""
        backend = _populated_backend()
        results = backend.delete_records("Leads", ["L001"])
        self.assertEqual(results[0]["status"], "success")
        records = backend.get_records("Leads")
        self.assertEqual(len(records), 2)  # L002, L003 remain

    def test_delete_nonexistent(self):
        """Test deleting non-existent record."""
        backend = _populated_backend()
        results = backend.delete_records("Leads", ["L999"])
        self.assertEqual(results[0]["status"], "error")

    def test_upsert_insert(self):
        """Test upsert creates new record."""
        backend = InMemoryCrmBackend()
        results = backend.upsert_records("Leads", [
            {"Last_Name": "New", "Email": "new@test.com"},
        ])
        self.assertEqual(results[0]["action"], "insert")
        records = backend.get_records("Leads")
        self.assertEqual(len(records), 1)

    def test_upsert_update(self):
        """Test upsert updates existing record on duplicate check."""
        backend = _populated_backend()
        results = backend.upsert_records("Leads", [
            {"Last_Name": "Smith", "Email": "updated@test.com"},
        ], duplicate_check_fields=["Last_Name"])
        self.assertEqual(results[0]["action"], "update")
        records = backend.get_records("Leads", ids=["L001"])
        self.assertEqual(records[0]["Email"], "updated@test.com")


class TestCrmRecord(unittest.TestCase):
    """Tests for CrmRecord dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        rec = CrmRecord(id="R1", module="Leads", data={"Name": "Test"})
        d = rec.to_dict()
        self.assertEqual(d["id"], "R1")
        self.assertEqual(d["Name"], "Test")

    def test_from_dict(self):
        """Test deserialization."""
        rec = CrmRecord.from_dict({"id": "R2", "Name": "Test2"}, module="Contacts")
        self.assertEqual(rec.id, "R2")
        self.assertEqual(rec.module, "Contacts")
        self.assertEqual(rec.data["Name"], "Test2")


# ═══════════════════════════════════════════════════════════════════════
# CrmGetRecordsTool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCrmGetRecordsTool(unittest.TestCase):
    """Tests for CrmGetRecordsTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = _populated_backend()
        self.tool = CrmGetRecordsTool(crm_backend=self.backend)

    def test_no_module(self):
        """Test error when no module."""
        result = run(self.tool.execute())
        self.assertFalse(result.success)

    def test_get_all(self):
        """Test getting all records."""
        result = run(self.tool.execute(module="Leads"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["record_count"], 3)

    def test_get_by_ids(self):
        """Test getting specific records."""
        result = run(self.tool.execute(module="Leads", ids="L001,L002"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["record_count"], 2)

    def test_get_with_fields(self):
        """Test field selection."""
        result = run(self.tool.execute(module="Leads", fields="Last_Name"))
        self.assertTrue(result.success)
        for rec in result.metadata["records"]:
            self.assertIn("Last_Name", rec)

    def test_empty_module(self):
        """Test empty module returns no records."""
        result = run(self.tool.execute(module="Deals"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["record_count"], 0)

    def test_pagination(self):
        """Test pagination."""
        result = run(self.tool.execute(module="Leads", per_page=2))
        self.assertEqual(result.metadata["record_count"], 2)

    def test_output_format(self):
        """Test output contains record info."""
        result = run(self.tool.execute(module="Leads"))
        self.assertIn("Smith", result.output)
        self.assertIn("L001", result.output)

    def test_tool_name(self):
        """Test tool name."""
        self.assertEqual(self.tool.name, "crm_get_records")


# ═══════════════════════════════════════════════════════════════════════
# CrmSearchRecordsTool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCrmSearchRecordsTool(unittest.TestCase):
    """Tests for CrmSearchRecordsTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = _populated_backend()
        self.tool = CrmSearchRecordsTool(crm_backend=self.backend)

    def test_no_module(self):
        """Test error when no module."""
        result = run(self.tool.execute(word="test"))
        self.assertFalse(result.success)

    def test_no_search_param(self):
        """Test error when no search parameter."""
        result = run(self.tool.execute(module="Leads"))
        self.assertFalse(result.success)
        self.assertIn("At least one", result.error)

    def test_search_by_word(self):
        """Test word search."""
        result = run(self.tool.execute(module="Leads", word="Smith"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["record_count"], 1)

    def test_search_by_email(self):
        """Test email search."""
        result = run(self.tool.execute(module="Leads", email="example.com"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["record_count"], 2)

    def test_search_by_phone(self):
        """Test phone search."""
        result = run(self.tool.execute(module="Leads", phone="555-0102"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["record_count"], 1)

    def test_search_no_results(self):
        """Test search with no matches."""
        result = run(self.tool.execute(module="Leads", word="zzzzz"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["record_count"], 0)

    def test_tool_name(self):
        """Test tool name."""
        self.assertEqual(self.tool.name, "crm_search_records")


# ═══════════════════════════════════════════════════════════════════════
# CrmCreateRecordsTool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCrmCreateRecordsTool(unittest.TestCase):
    """Tests for CrmCreateRecordsTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = InMemoryCrmBackend()
        self.tool = CrmCreateRecordsTool(crm_backend=self.backend)

    def test_no_module(self):
        """Test error when no module."""
        result = run(self.tool.execute(data=[{"Name": "Test"}]))
        self.assertFalse(result.success)

    def test_no_data(self):
        """Test error when no data."""
        result = run(self.tool.execute(module="Leads"))
        self.assertFalse(result.success)

    def test_empty_data(self):
        """Test error when empty data list."""
        result = run(self.tool.execute(module="Leads", data=[]))
        self.assertFalse(result.success)

    def test_invalid_data_type(self):
        """Test error when data is not a list."""
        result = run(self.tool.execute(module="Leads", data="not a list"))
        self.assertFalse(result.success)

    def test_create_single(self):
        """Test creating a single record."""
        result = run(self.tool.execute(module="Leads", data=[
            {"Last_Name": "NewLead", "Email": "new@test.com"},
        ]))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["created_count"], 1)

    def test_create_multiple(self):
        """Test creating multiple records."""
        result = run(self.tool.execute(module="Leads", data=[
            {"Last_Name": "Lead1"},
            {"Last_Name": "Lead2"},
            {"Last_Name": "Lead3"},
        ]))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["created_count"], 3)

    def test_create_verifiable(self):
        """Test created records are retrievable."""
        run(self.tool.execute(module="Leads", data=[
            {"id": "NEW1", "Last_Name": "Verifiable"},
        ]))
        records = self.backend.get_records("Leads", ids=["NEW1"])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["Last_Name"], "Verifiable")

    def test_tool_name(self):
        """Test tool name."""
        self.assertEqual(self.tool.name, "crm_create_records")


# ═══════════════════════════════════════════════════════════════════════
# CrmUpdateRecordsTool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCrmUpdateRecordsTool(unittest.TestCase):
    """Tests for CrmUpdateRecordsTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = _populated_backend()
        self.tool = CrmUpdateRecordsTool(crm_backend=self.backend)

    def test_no_module(self):
        """Test error when no module."""
        result = run(self.tool.execute(data=[{"id": "L001", "Name": "X"}]))
        self.assertFalse(result.success)

    def test_no_data(self):
        """Test error when no data."""
        result = run(self.tool.execute(module="Leads"))
        self.assertFalse(result.success)

    def test_missing_id(self):
        """Test error when record has no ID."""
        result = run(self.tool.execute(module="Leads", data=[{"Name": "NoID"}]))
        self.assertFalse(result.success)
        self.assertIn("id", result.error)

    def test_update_success(self):
        """Test successful update."""
        result = run(self.tool.execute(module="Leads", data=[
            {"id": "L001", "Last_Name": "Smith-Updated"},
        ]))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["updated_count"], 1)

    def test_update_nonexistent(self):
        """Test updating non-existent record."""
        result = run(self.tool.execute(module="Leads", data=[
            {"id": "L999", "Last_Name": "Ghost"},
        ]))
        self.assertTrue(result.success)  # Tool succeeds, but count is 0
        self.assertEqual(result.metadata["updated_count"], 0)

    def test_update_verifiable(self):
        """Test updated data persists."""
        run(self.tool.execute(module="Leads", data=[
            {"id": "L002", "Last_Name": "Jones-Updated"},
        ]))
        records = self.backend.get_records("Leads", ids=["L002"])
        self.assertEqual(records[0]["Last_Name"], "Jones-Updated")

    def test_tool_name(self):
        """Test tool name."""
        self.assertEqual(self.tool.name, "crm_update_records")


# ═══════════════════════════════════════════════════════════════════════
# CrmDeleteRecordsTool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCrmDeleteRecordsTool(unittest.TestCase):
    """Tests for CrmDeleteRecordsTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = _populated_backend()
        self.tool = CrmDeleteRecordsTool(crm_backend=self.backend)

    def test_no_module(self):
        """Test error when no module."""
        result = run(self.tool.execute(ids="L001"))
        self.assertFalse(result.success)

    def test_no_ids(self):
        """Test error when no IDs."""
        result = run(self.tool.execute(module="Leads"))
        self.assertFalse(result.success)

    def test_delete_single(self):
        """Test deleting a single record."""
        result = run(self.tool.execute(module="Leads", ids="L001"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["deleted_count"], 1)

    def test_delete_multiple(self):
        """Test deleting multiple records."""
        result = run(self.tool.execute(module="Leads", ids="L001,L002"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["deleted_count"], 2)

    def test_delete_nonexistent(self):
        """Test deleting non-existent record."""
        result = run(self.tool.execute(module="Leads", ids="L999"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["deleted_count"], 0)

    def test_delete_verifiable(self):
        """Test deleted records are gone."""
        run(self.tool.execute(module="Leads", ids="L001"))
        records = self.backend.get_records("Leads")
        ids = [r["id"] for r in records]
        self.assertNotIn("L001", ids)

    def test_tool_name(self):
        """Test tool name."""
        self.assertEqual(self.tool.name, "crm_delete_records")


# ═══════════════════════════════════════════════════════════════════════
# CrmUpsertRecordsTool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCrmUpsertRecordsTool(unittest.TestCase):
    """Tests for CrmUpsertRecordsTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = _populated_backend()
        self.tool = CrmUpsertRecordsTool(crm_backend=self.backend)

    def test_no_module(self):
        """Test error when no module."""
        result = run(self.tool.execute(data=[{"Name": "X"}]))
        self.assertFalse(result.success)

    def test_no_data(self):
        """Test error when no data."""
        result = run(self.tool.execute(module="Leads"))
        self.assertFalse(result.success)

    def test_upsert_insert(self):
        """Test upsert creating a new record."""
        result = run(self.tool.execute(module="Leads", data=[
            {"Last_Name": "Brand New", "Email": "new@test.com"},
        ]))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["insert_count"], 1)
        self.assertEqual(result.metadata["update_count"], 0)

    def test_upsert_update(self):
        """Test upsert updating an existing record."""
        result = run(self.tool.execute(module="Leads", data=[
            {"Last_Name": "Smith", "Email": "updated@test.com"},
        ], duplicate_check_fields=["Last_Name"]))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["update_count"], 1)
        self.assertEqual(result.metadata["insert_count"], 0)

    def test_upsert_mixed(self):
        """Test upsert with mix of inserts and updates."""
        result = run(self.tool.execute(module="Leads", data=[
            {"Last_Name": "Smith", "Email": "u1@test.com"},
            {"Last_Name": "NewPerson", "Email": "u2@test.com"},
        ], duplicate_check_fields=["Last_Name"]))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["update_count"], 1)
        self.assertEqual(result.metadata["insert_count"], 1)

    def test_upsert_no_dup_fields(self):
        """Test upsert without duplicate check fields inserts all."""
        result = run(self.tool.execute(module="Leads", data=[
            {"Last_Name": "Smith"},
        ]))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["insert_count"], 1)

    def test_tool_name(self):
        """Test tool name."""
        self.assertEqual(self.tool.name, "crm_upsert_records")


# ═══════════════════════════════════════════════════════════════════════
# Tool Schemas
# ═══════════════════════════════════════════════════════════════════════


class TestToolSchemas(unittest.TestCase):
    """Tests for tool schema validation."""

    def test_all_tools_have_schemas(self):
        """All CRM tools have valid input_schema."""
        tools = [
            CrmGetRecordsTool(),
            CrmSearchRecordsTool(),
            CrmCreateRecordsTool(),
            CrmUpdateRecordsTool(),
            CrmDeleteRecordsTool(),
            CrmUpsertRecordsTool(),
        ]
        for tool in tools:
            self.assertIsInstance(tool.input_schema, dict, f"{tool.name} missing schema")
            self.assertEqual(tool.input_schema["type"], "object")

    def test_tool_names_unique(self):
        """All tool names are unique."""
        tools = [
            CrmGetRecordsTool(),
            CrmSearchRecordsTool(),
            CrmCreateRecordsTool(),
            CrmUpdateRecordsTool(),
            CrmDeleteRecordsTool(),
            CrmUpsertRecordsTool(),
        ]
        names = [t.name for t in tools]
        self.assertEqual(len(names), len(set(names)))

    def test_expected_tool_names(self):
        """Verify the expected set of tool names."""
        tools = [
            CrmGetRecordsTool(),
            CrmSearchRecordsTool(),
            CrmCreateRecordsTool(),
            CrmUpdateRecordsTool(),
            CrmDeleteRecordsTool(),
            CrmUpsertRecordsTool(),
        ]
        names = {t.name for t in tools}
        expected = {
            "crm_get_records", "crm_search_records",
            "crm_create_records", "crm_update_records",
            "crm_delete_records", "crm_upsert_records",
        }
        self.assertEqual(names, expected)


# ═══════════════════════════════════════════════════════════════════════
# Main Wiring
# ═══════════════════════════════════════════════════════════════════════


class TestMainWiring(unittest.TestCase):
    """Tests for main.py wiring of Sprint 36 tools."""

    def test_crm_wiring(self):
        """Verify Sprint 36 tools are wired in main.py."""
        import inspect
        from cowork_agent import main as main_mod

        source = inspect.getsource(main_mod)
        self.assertIn("CrmGetRecordsTool", source)
        self.assertIn("CrmSearchRecordsTool", source)
        self.assertIn("CrmCreateRecordsTool", source)
        self.assertIn("CrmUpdateRecordsTool", source)
        self.assertIn("CrmDeleteRecordsTool", source)
        self.assertIn("CrmUpsertRecordsTool", source)
        self.assertIn("crm_tools", source)


# ═══════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases(unittest.TestCase):
    """Edge case tests for Sprint 36 tools."""

    def test_standard_modules_constant(self):
        """Test STANDARD_MODULES has expected modules."""
        self.assertIn("Leads", STANDARD_MODULES)
        self.assertIn("Contacts", STANDARD_MODULES)
        self.assertIn("Deals", STANDARD_MODULES)
        self.assertIn("Accounts", STANDARD_MODULES)
        self.assertGreater(len(STANDARD_MODULES), 15)

    def test_crud_lifecycle(self):
        """Test complete CRUD lifecycle."""
        backend = InMemoryCrmBackend()
        create_tool = CrmCreateRecordsTool(crm_backend=backend)
        get_tool = CrmGetRecordsTool(crm_backend=backend)
        update_tool = CrmUpdateRecordsTool(crm_backend=backend)
        delete_tool = CrmDeleteRecordsTool(crm_backend=backend)

        # Create
        r = run(create_tool.execute(module="Leads", data=[
            {"id": "LIFE1", "Last_Name": "Lifecycle", "Email": "lc@test.com"},
        ]))
        self.assertTrue(r.success)

        # Read
        r = run(get_tool.execute(module="Leads", ids="LIFE1"))
        self.assertEqual(r.metadata["record_count"], 1)

        # Update
        r = run(update_tool.execute(module="Leads", data=[
            {"id": "LIFE1", "Last_Name": "Updated"},
        ]))
        self.assertTrue(r.success)

        # Verify update
        r = run(get_tool.execute(module="Leads", ids="LIFE1"))
        self.assertEqual(r.metadata["records"][0]["Last_Name"], "Updated")

        # Delete
        r = run(delete_tool.execute(module="Leads", ids="LIFE1"))
        self.assertTrue(r.success)

        # Verify delete
        r = run(get_tool.execute(module="Leads", ids="LIFE1"))
        self.assertEqual(r.metadata["record_count"], 0)

    def test_shared_backend(self):
        """Test multiple tools sharing the same backend."""
        backend = InMemoryCrmBackend()
        create = CrmCreateRecordsTool(crm_backend=backend)
        search = CrmSearchRecordsTool(crm_backend=backend)

        run(create.execute(module="Contacts", data=[
            {"id": "SH1", "Last_Name": "Shared", "Email": "shared@test.com"},
        ]))

        r = run(search.execute(module="Contacts", word="Shared"))
        self.assertEqual(r.metadata["record_count"], 1)

    def test_cross_module_isolation(self):
        """Test modules are isolated from each other."""
        backend = InMemoryCrmBackend()
        create = CrmCreateRecordsTool(crm_backend=backend)
        get = CrmGetRecordsTool(crm_backend=backend)

        run(create.execute(module="Leads", data=[{"id": "X1", "Name": "Lead"}]))
        run(create.execute(module="Contacts", data=[{"id": "X2", "Name": "Contact"}]))

        leads = run(get.execute(module="Leads"))
        contacts = run(get.execute(module="Contacts"))
        self.assertEqual(leads.metadata["record_count"], 1)
        self.assertEqual(contacts.metadata["record_count"], 1)

    def test_delete_idempotent(self):
        """Test deleting already-deleted record returns error."""
        backend = _populated_backend()
        tool = CrmDeleteRecordsTool(crm_backend=backend)

        run(tool.execute(module="Leads", ids="L001"))
        r = run(tool.execute(module="Leads", ids="L001"))
        self.assertEqual(r.metadata["deleted_count"], 0)

    def test_default_backend(self):
        """Test tools create their own backend if none provided."""
        tool = CrmGetRecordsTool()
        result = run(tool.execute(module="Leads"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["record_count"], 0)

    def test_sort_by_modified_time(self):
        """Test sorting by Modified_Time."""
        backend = _populated_backend()
        records = backend.get_records("Leads", sort_by="Modified_Time", sort_order="asc")
        self.assertEqual(len(records), 3)


if __name__ == "__main__":
    unittest.main()
