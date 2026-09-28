"""Read a match store with DuckDB over its parquet tables."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from regista.store.build import STORE_FORMAT, config_hash


class StoreError(RuntimeError):
    """Raised when a store is missing, malformed, or built with different parameters."""


class Store:
    """One match's store. Tables are exposed as DuckDB views named after the parquet files."""

    def __init__(self, path: Path, params: dict | None = None, calibration: dict | None = None):
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise StoreError(f"no store at {self.path}; run `regista build-store` first")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("store_format") != STORE_FORMAT:
            raise StoreError(f"{self.path}: store format {self.manifest.get('store_format')}, "
                             f"expected {STORE_FORMAT}; rebuild it")  # fmt: skip
        if params is not None and calibration is not None:
            expected = config_hash(params, calibration)
            built = self.manifest["config_hash"]
            if built != expected:
                raise StoreError(
                    f"{self.path} was built with different parameters "
                    f"({built} != {expected}); rebuild it"
                )
        self.con = duckdb.connect()
        for name in self.manifest["tables"]:
            file = self.path / f"{name}.parquet"
            if not file.exists():
                raise StoreError(f"{self.path}: table {name} listed in manifest but missing")
            self.con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{file}')")

    @property
    def match_id(self) -> str:
        return self.manifest["match_id"]

    @property
    def source(self) -> str:
        return self.manifest["source"]

    def sql(self, query: str, params: list | None = None) -> pd.DataFrame:
        """Run a read-only SQL query against the store's views."""
        return self.con.execute(query, params or []).df()

    def table(self, name: str) -> pd.DataFrame:
        if name not in self.manifest["tables"]:
            raise StoreError(f"unknown table {name!r}; have {sorted(self.manifest['tables'])}")
        return pd.read_parquet(self.path / f"{name}.parquet")

    def close(self) -> None:
        self.con.close()
