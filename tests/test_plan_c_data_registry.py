"""Tests for Plan C source registry and deterministic fetch planning."""

from __future__ import annotations

import json
import unittest

from data import registry
from data.contract import hash_payload, source_supervision_role


class TestPlanCSourceRegistry(unittest.TestCase):
    def _descriptor_kwargs(self, source: str) -> dict:
        resolved = registry.get_source_descriptor(source)
        return dict(
            source=resolved.source,
            supervision_status=resolved.supervision_status,
            source_urls=resolved.source_urls,
            request_headers=resolved.request_headers,
            required_row_fields=resolved.required_row_fields,
            optional_row_fields=resolved.optional_row_fields,
            row_parser_hint=resolved.row_parser_hint,
            schema_version=resolved.schema_version,
            source_aliases=resolved.source_aliases,
        )

    def _manual_fetch_plan_hash(self, descriptor: registry.SourceDescriptor) -> str:
        plan = {
            "registry_version": registry.REGISTRY_VERSION,
            "schema_version": "1",
            "sources": [descriptor.as_plan_payload()],
            "source_count": 1,
        }
        plan["fetch_plan_hash"] = hash_payload(plan)
        return plan["fetch_plan_hash"]

    def test_source_list_is_stable_and_sorted(self) -> None:
        sources = registry.list_sources()
        self.assertEqual(sources, tuple(sorted(sources)))
        self.assertEqual(len(sources), len(set(sources)))
        self.assertGreaterEqual(len(sources), 7)

    def test_catalog_sources_match_contract_roles(self) -> None:
        for source in registry.list_sources():
            descriptor = registry.get_source_descriptor(source)
            self.assertEqual(descriptor.source, source)
            self.assertEqual(
                descriptor.supervision_status,
                source_supervision_role(source),
            )
            self.assertNotIn(source, descriptor.source_aliases)

    def test_figshare_agg_alias_family_is_deterministic(self) -> None:
        figshare = registry.get_source_descriptor("figshare_agg")
        self.assertEqual(figshare.source, "figshare_agg")
        for alias in ("figshare_a3d", "figshare_a3", "figshareagg", "figshare-a3d"):
            with self.subTest(alias=alias):
                resolved = registry.get_source_descriptor(alias)
                self.assertEqual(resolved.source, "figshare_agg")
                self.assertEqual(resolved.supervision_status, figshare.supervision_status)

        self.assertNotIn("figshare_agg", figshare.source_aliases)

    def test_alias_parity_and_exclusions(self) -> None:
        alias_map = {
            "flab": ("flab",),
            "proteingym": ("proteingym", "proteingym_dms", "protein-gym"),
            "canya": ("canya",),
            "figshare_agg": (
                "figshare_agg",
                "figshare_a3d",
                "figshare_a3",
                "figshareagg",
                "figshare-a3d",
            ),
            "abdev": ("abdev", "ab-dev"),
            "antiref": ("antiref",),
            "sabdab": ("sabdab",),
            "pdb_anchor": ("pdb_anchor", "anchors", "anchor", "pdb-anchor"),
        }
        for canonical, aliases in alias_map.items():
            for alias in aliases:
                with self.subTest(canonical=canonical, alias=alias):
                    resolved = registry.get_source_descriptor(alias)
                    self.assertEqual(resolved.source, canonical)
                    self.assertEqual(
                        resolved.supervision_status, source_supervision_role(canonical)
                    )

        for excluded in ("unknown", "legacy", "input", "benchmark"):
            with self.subTest(excluded=excluded):
                with self.assertRaisesRegex(
                    ValueError, r"unknown source .*known sources are"
                ):
                    registry.get_source_descriptor(excluded)

    def test_aliases_do_not_collide(self) -> None:
        alias_targets: dict[str, str] = {}
        for source in registry.list_sources():
            descriptor = registry.get_source_descriptor(source)
            for alias in descriptor.source_aliases:
                with self.subTest(source=source, alias=alias):
                    if alias in alias_targets:
                        self.assertEqual(alias_targets[alias], source)
                    alias_targets[alias] = source
        for alias, mapped_source in alias_targets.items():
            self.assertEqual(
                registry.get_source_descriptor(alias).source,
                mapped_source,
            )

    def test_unknown_and_empty_sources_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"unknown source .*known sources are"):
            registry.get_source_descriptor("not-a-source")

        with self.assertRaisesRegex(ValueError, r"invalid source name:"):
            registry.get_source_descriptor("   ")

    def test_normalize_source_filter_rejects_bad_types(self) -> None:
        with self.assertRaisesRegex(TypeError, "source names must be strings"):
            registry.build_fetch_plan([12, "flab"])  # type: ignore[arg-type]

        with self.assertRaisesRegex(
            TypeError, "source_filter must be a string or sequence"
        ):
            registry.build_fetch_plan(123)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "source_filter cannot be empty"):
            registry.build_fetch_plan([])

    def test_build_fetch_plan_is_deterministic(self) -> None:
        first = registry.build_fetch_plan(["abdev", "flab", "abdev", "canya"])
        second = registry.build_fetch_plan(["canya", "flab", "abdev"])
        self.assertEqual(first["sources"], second["sources"])
        self.assertEqual(first["fetch_plan_hash"], second["fetch_plan_hash"])
        self.assertEqual(
            [source["source"] for source in first["sources"]],
            ["abdev", "canya", "flab"],
        )
        self.assertEqual(
            registry.fetch_plan_to_json(first),
            registry.fetch_plan_to_json(second),
        )

    def test_fetch_plan_inputs_are_permutation_safe_and_stable(self) -> None:
        explicit_all = registry.build_fetch_plan("all")
        wildcard = registry.build_fetch_plan("*")
        none = registry.build_fetch_plan(None)
        self.assertEqual(explicit_all["sources"], wildcard["sources"])
        self.assertEqual(explicit_all["sources"], none["sources"])
        self.assertEqual(explicit_all["fetch_plan_hash"], wildcard["fetch_plan_hash"])
        self.assertEqual(explicit_all["fetch_plan_hash"], none["fetch_plan_hash"])

        loaded = json.loads(registry.fetch_plan_to_json(explicit_all))
        self.assertEqual(loaded["source_count"], len(loaded["sources"]))

    def test_plan_contains_required_fields(self) -> None:
        plan = registry.build_fetch_plan("flab")
        plan_sources = plan["sources"]
        self.assertEqual(len(plan_sources), 1)
        source_plan = plan_sources[0]

        expected_fields = {
            "source",
            "supervision_status",
            "schema_version",
            "source_urls",
            "request_headers",
            "required_row_fields",
            "optional_row_fields",
            "row_parser_hint",
        }
        self.assertTrue(expected_fields.issubset(source_plan))
        self.assertGreater(len(source_plan["source_urls"]), 0)
        self.assertTrue(all(isinstance(url, str) for url in source_plan["source_urls"]))
        self.assertIn("variant_sequence", source_plan["required_row_fields"])
        self.assertIn("source", source_plan["required_row_fields"])
        self.assertIn("endpoint_value", source_plan["required_row_fields"])

    def test_request_headers_are_deterministically_sorted(self) -> None:
        plan = registry.build_fetch_plan("flab")
        headers = plan["sources"][0]["request_headers"]
        self.assertEqual(headers, sorted(headers))

    def test_invalid_direct_descriptor_is_rejected_for_schema_mismatches(self) -> None:
        valid = self._descriptor_kwargs("flab")

        with self.assertRaises(ValueError):
            registry.SourceDescriptor(
                source=1,  # type: ignore[arg-type]
                supervision_status="supervised",
                source_urls=(),
                request_headers=(),
                required_row_fields=("source",),
                optional_row_fields=(),
                row_parser_hint="ok",
            )

        with self.assertRaises(ValueError):
            registry.SourceDescriptor(
                **{**valid, "supervision_status": "background_ood"}
            )

        with self.assertRaises(ValueError):
            registry.SourceDescriptor(
                **{**valid, "required_row_fields": ["source", "variant_sequence"]}  # type: ignore[arg-type]
            )

        with self.assertRaises(ValueError):
            registry.SourceDescriptor(
                **{**valid, "optional_row_fields": ["source", "variant_sequence"]}  # type: ignore[arg-type]
            )

        with self.assertRaises(ValueError):
            registry.SourceDescriptor(
                **{**valid, "required_row_fields": ("source", "label"), "optional_row_fields": ("label",)}
            )

        with self.assertRaises(ValueError):
            registry.SourceDescriptor(
                **{**valid, "required_row_fields": ("source", 1)}  # type: ignore[arg-type]
            )

        with self.assertRaises(ValueError):
            registry.SourceDescriptor(
                **{**valid, "source_aliases": ("flab",)}
            )

        with self.assertRaises(ValueError):
            registry.SourceDescriptor(
                **{**valid, "row_parser_hint": "   "}
            )

    def test_source_urls_validation_and_edge_cases(self) -> None:
        valid = self._descriptor_kwargs("flab")
        anchor = self._descriptor_kwargs("pdb_anchor")

        with self.assertRaisesRegex(ValueError, "source_urls must be a tuple"):
            registry.SourceDescriptor(**{**valid, "source_urls": 1})

        with self.assertRaisesRegex(ValueError, "source_urls must be a tuple"):
            registry.SourceDescriptor(**{**valid, "source_urls": ["https://example.com"]})

        with self.assertRaisesRegex(ValueError, "source_urls must be a tuple"):
            registry.SourceDescriptor(
                **{**valid, "source_urls": "https://example.com"}  # type: ignore[arg-type]
            )

        descriptor = registry.SourceDescriptor(**{**anchor, "source_urls": ()})
        self.assertEqual(descriptor.source, "pdb_anchor")
        self.assertEqual(descriptor.source_urls, ())

        with self.assertRaisesRegex(ValueError, "source URL"):
            registry.SourceDescriptor(
                **{**valid, "source_urls": ("https://example.com", 1)}  # type: ignore[arg-type]
            )

        normalized = registry.SourceDescriptor(
            source=valid["source"],
            supervision_status=valid["supervision_status"],
            source_urls=("https://example.com/example.tsv",),
            request_headers=valid["request_headers"],
            required_row_fields=valid["required_row_fields"],
            optional_row_fields=valid["optional_row_fields"],
            row_parser_hint=valid["row_parser_hint"],
            schema_version=valid["schema_version"],
            source_aliases=valid["source_aliases"],
        )
        self.assertEqual(normalized.source_urls, ("https://example.com/example.tsv",))

    def test_row_schema_hash_ignores_transport_and_parser(self) -> None:
        base = registry.get_source_descriptor("flab")
        alt_urls = registry.SourceDescriptor(
            source=base.source,
            supervision_status=base.supervision_status,
            source_urls=base.source_urls + ("https://example.com/flab-extra",),
            request_headers=(("User-Agent", "unit-test"),),
            required_row_fields=base.required_row_fields,
            optional_row_fields=base.optional_row_fields,
            row_parser_hint="alternative parser prose",
            schema_version=base.schema_version,
            source_aliases=base.source_aliases,
        )

        self.assertEqual(base.row_schema_hash(), alt_urls.row_schema_hash())
        self.assertNotEqual(base.descriptor_hash(), alt_urls.descriptor_hash())
        self.assertNotEqual(
            self._manual_fetch_plan_hash(base),
            self._manual_fetch_plan_hash(alt_urls),
        )

    def test_row_schema_hash_is_order_invariant_for_field_sets(self) -> None:
        base = registry.get_source_descriptor("flab")
        reordered_required = tuple(reversed(base.required_row_fields))
        reordered_optional = tuple(reversed(base.optional_row_fields))
        reordered = registry.SourceDescriptor(
            source=base.source,
            supervision_status=base.supervision_status,
            source_urls=base.source_urls,
            request_headers=base.request_headers,
            required_row_fields=reordered_required,
            optional_row_fields=reordered_optional,
            row_parser_hint=base.row_parser_hint,
            schema_version=base.schema_version,
            source_aliases=base.source_aliases,
        )
        self.assertEqual(base.row_schema_hash(), reordered.row_schema_hash())
        self.assertEqual(base.descriptor_hash(), reordered.descriptor_hash())

    def test_row_schema_hash_sensitive_to_semantic_changes(self) -> None:
        base = registry.get_source_descriptor("flab")
        fields = base.required_row_fields + ("new_aux_field",)
        mutated_fields = registry.SourceDescriptor(
            source=base.source,
            supervision_status=base.supervision_status,
            source_urls=base.source_urls,
            request_headers=base.request_headers,
            required_row_fields=fields,
            optional_row_fields=base.optional_row_fields,
            row_parser_hint=base.row_parser_hint,
            schema_version=base.schema_version,
            source_aliases=base.source_aliases,
        )
        self.assertNotEqual(base.row_schema_hash(), mutated_fields.row_schema_hash())

        changed_schema_version = registry.SourceDescriptor(
            source=base.source,
            supervision_status=base.supervision_status,
            source_urls=base.source_urls,
            request_headers=base.request_headers,
            required_row_fields=base.required_row_fields,
            optional_row_fields=base.optional_row_fields,
            row_parser_hint=base.row_parser_hint,
            schema_version="2",
            source_aliases=base.source_aliases,
        )
        self.assertNotEqual(base.row_schema_hash(), changed_schema_version.row_schema_hash())

        other_role_descriptor = registry.get_source_descriptor("proteingym")
        self.assertNotEqual(base.row_schema_hash(), other_role_descriptor.row_schema_hash())
