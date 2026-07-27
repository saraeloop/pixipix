"""Compatibility facade for shared downstream pipeline lifecycles."""

from ..pipeline.input import (  # noqa: I001 - keep deliberate compatibility re-exports grouped
    InputStageFrame as InputStageFrame,
    LoadedStageInput as LoadedStageInput,
    ValidatedStageFrame as ValidatedStageFrame,
    ValidatedStageInput as ValidatedStageInput,
    decode_stage_input as decode_stage_input,
    load_stage_input as load_stage_input,
    validate_stage_input as validate_stage_input,
)
from ..pipeline.publication import (
    OutputFrameImage as OutputFrameImage,
    _valid_owned_output as _valid_owned_output,
    publish_stage_output as publish_stage_output,
    validate_stage_output_target as validate_stage_output_target,
)
