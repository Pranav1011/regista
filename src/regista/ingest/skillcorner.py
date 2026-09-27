"""SkillCorner open data (broadcast tracking, MIT), via kloppy's open-data loader.

Data: https://github.com/SkillCorner/opendata. Credit SkillCorner.
Positions come from broadcast video: players off camera are extrapolated by the
provider. Each match's real pitch size is rescaled onto the canonical 105 x 68 m.
"""

from __future__ import annotations

from kloppy import skillcorner

from regista.ingest._kloppy import IngestResult, dataset_to_canonical
from regista.schema import Source


def load(match_id: str) -> IngestResult:
    """Download one open match and convert it to canonical tables."""
    dataset = skillcorner.load_open_data(match_id=match_id)
    return dataset_to_canonical(dataset, match_id, Source.SKILLCORNER)
