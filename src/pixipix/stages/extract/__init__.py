"""Compatibility facade for deterministic source-space component extraction."""

# Same-name aliases are explicit mypy re-exports; keep owner groups intact.
# isort: off
from .analysis import (
    ComponentMap as ComponentMap,
    filter_components as filter_components,
    label_components as label_components,
    order_components as order_components,
)
from .api import (
    extract_source as extract_source,
    inspect_source as inspect_source,
)
from .planning import (
    project_extract_resources as project_extract_resources,
    project_extracted_frames as project_extracted_frames,
)
from .publication import publish_extraction as publish_extraction
# isort: on
