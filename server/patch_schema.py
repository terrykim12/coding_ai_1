from pydantic import BaseModel, Field
from typing import List, Union, Optional, Literal
from enum import Enum

class LocationType(str, Enum):
    ANCHOR = "anchor"
    RANGE = "range"
    REGEX = "regex"
    AST = "ast"

class ActionType(str, Enum):
    INSERT_BEFORE = "insert_before"
    INSERT_AFTER = "insert_after"
    INSERT_AFTER_BLOCK = "insert_after_block"
    REPLACE_RANGE = "replace_range"
    DELETE_RANGE = "delete_range"

class Position(BaseModel):
    line: int = Field(..., ge=0)
    col: int = Field(..., ge=0)

class Range(BaseModel):
    start: Position
    end: Position

class AnchorLocation(BaseModel):
    type: Literal[LocationType.ANCHOR] = LocationType.ANCHOR
    before: Optional[str] = None
    after: Optional[str] = None

class RangeLocation(BaseModel):
    type: Literal[LocationType.RANGE] = LocationType.RANGE
    range: Range

class RegexLocation(BaseModel):
    type: Literal[LocationType.REGEX] = LocationType.REGEX
    pattern: str

class ASTLocation(BaseModel):
    type: Literal[LocationType.AST] = LocationType.AST
    node_type: str
    selector: str

Location = Union[AnchorLocation, RangeLocation, RegexLocation, ASTLocation]

class Edit(BaseModel):
    path: str
    loc: Location
    action: ActionType
    range: Optional[Range] = None  # For range-based actions
    code: str

class PatchJSON(BaseModel):
    version: str = "1"
    edits: List[Edit]
    metadata: Optional[dict] = None

# Validation helpers
def validate_patch(patch_data: dict) -> PatchJSON:
    """Validate and return PatchJSON object"""
    return PatchJSON(**patch_data)

def validate_edit(edit_data: dict) -> Edit:
    """Validate and return Edit object"""
    return Edit(**edit_data)

