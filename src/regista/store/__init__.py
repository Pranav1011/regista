"""Phase 2 match store: everything the agent, tools, and viewer need, precomputed per match."""

from regista.store.build import build_store
from regista.store.reader import Store

__all__ = ["Store", "build_store"]
