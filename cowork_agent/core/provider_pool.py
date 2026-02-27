"""
Provider Pool — Manages multiple LLM providers as a unified pool.

Supports tier-based selection, lazy initialization, hot-swapping,
and health-aware provider routing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from .model_router import ModelTier
from .provider_health_tracker import ProviderHealthTracker


@dataclass
class ProviderEntry:
    """A registered provider in the pool."""
    name: str
    provider: object                    # BaseLLMProvider instance
    tier: ModelTier = ModelTier.BALANCED
    initialized: bool = True
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        provider_name = ""
        if hasattr(self.provider, "provider_name"):
            provider_name = self.provider.provider_name
        model = ""
        if hasattr(self.provider, "model"):
            model = self.provider.model
        return {
            "name": self.name,
            "provider_class": provider_name,
            "model": model,
            "tier": self.tier.value,
            "initialized": self.initialized,
        }


class ProviderPool:
    """
    Manages multiple LLM providers as a pool.

    Features:
        - Tier-based provider selection (FAST/BALANCED/POWERFUL)
        - Health-aware routing (prefers healthier providers within a tier)
        - Hot-swapping of the active default provider
        - Lazy initialization support
    """

    def __init__(
        self,
        health_tracker: Optional[ProviderHealthTracker] = None,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._entries: dict[str, ProviderEntry] = {}
        self._health_tracker = health_tracker
        self._active_name: Optional[str] = None

    # ── Registration ──────────────────────────────────────────────

    def register(
        self,
        name: str,
        provider: object,
        tier: ModelTier = ModelTier.BALANCED,
        config: Optional[dict] = None,
    ) -> None:
        """Register a provider in the pool."""
        entry = ProviderEntry(
            name=name,
            provider=provider,
            tier=tier,
            initialized=True,
            config=config or {},
        )
        self._entries[name] = entry
        # First registered provider becomes the default
        if self._active_name is None:
            self._active_name = name

    # ── Lookup ────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[object]:
        """Get a provider by name."""
        entry = self._entries.get(name)
        return entry.provider if entry else None

    def get_entry(self, name: str) -> Optional[ProviderEntry]:
        """Get the full ProviderEntry by name."""
        return self._entries.get(name)

    @property
    def active(self) -> Optional[object]:
        """Get the current active (default) provider."""
        if self._active_name and self._active_name in self._entries:
            return self._entries[self._active_name].provider
        return None

    @property
    def active_name(self) -> Optional[str]:
        return self._active_name

    # ── Tier-based selection ──────────────────────────────────────

    def get_for_tier(self, tier: ModelTier) -> Optional[object]:
        """
        Get the best provider for a given tier.

        If a health tracker is available, selects the healthiest provider
        within the tier. Otherwise returns the first match.
        """
        tier_entries = [
            e for e in self._entries.values()
            if e.tier == tier and e.initialized
        ]

        if not tier_entries:
            return None

        if len(tier_entries) == 1:
            return tier_entries[0].provider

        # Multiple providers for this tier — use health scores
        if self._health_tracker:
            best_entry = None
            best_score = -1.0
            for entry in tier_entries:
                score = self._health_tracker.get_score(entry.name).score
                if score > best_score:
                    best_score = score
                    best_entry = entry
            return best_entry.provider if best_entry else tier_entries[0].provider

        return tier_entries[0].provider

    def list_for_tier(self, tier: ModelTier) -> list[ProviderEntry]:
        """List all providers registered for a tier."""
        return [e for e in self._entries.values() if e.tier == tier]

    # ── Hot-swapping ──────────────────────────────────────────────

    def swap_active(self, name: str) -> bool:
        """
        Change the active (default) provider.

        Returns True if swap succeeded, False if provider not found.
        """
        if name not in self._entries:
            return False
        self._active_name = name
        return True

    # ── Listing ───────────────────────────────────────────────────

    def list_providers(self) -> list[ProviderEntry]:
        """List all registered providers."""
        return list(self._entries.values())

    @property
    def provider_names(self) -> list[str]:
        return list(self._entries.keys())

    @property
    def size(self) -> int:
        return len(self._entries)

    # ── Health check ──────────────────────────────────────────────

    async def health_check_all(self) -> dict[str, dict]:
        """
        Run health checks on all providers in parallel.

        Returns a dict of {name: health_result} for each provider.
        """
        results: dict[str, dict] = {}

        async def _check_one(name: str, entry: ProviderEntry):
            try:
                if hasattr(entry.provider, "health_check"):
                    result = await entry.provider.health_check()
                    results[name] = {"status": "ok", **result}
                else:
                    results[name] = {"status": "ok", "note": "no health_check method"}
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}

        await asyncio.gather(
            *(_check_one(n, e) for n, e in self._entries.items())
        )
        return results

    # ── Summary ───────────────────────────────────────────────────

    def summary(self) -> dict:
        """Pool status summary."""
        return {
            "enabled": self._enabled,
            "active": self._active_name,
            "provider_count": self.size,
            "providers": {
                name: entry.to_dict()
                for name, entry in self._entries.items()
            },
            "tiers": {
                tier.value: [e.name for e in self.list_for_tier(tier)]
                for tier in ModelTier
            },
        }
