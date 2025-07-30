from dataclasses import dataclass
from typing import Optional

from marshmallow import Schema, fields

from common_ml.types import Data

@dataclass
class UploadArgs(Data):
    aggregate: bool=False
    authorization: Optional[str]=None
    write_token: str=""

    # finalize args, set by default
    leave_open: bool=True
    force: bool=True
    replace: bool=True
    
    @staticmethod
    def from_dict(data: dict) -> 'UploadArgs':
        class UploadArgsSchema(Schema):
            aggregate = fields.Bool(required=False, missing=False)
            write_token = fields.Str(required=False, missing="")
            authorization = fields.Str(required=False, missing=None)
            
        return UploadArgs(**UploadArgsSchema().load(data))
    
@dataclass
class FinalizeArgs(Data):
    write_token: str
    replace: bool=False
    force: bool=False
    # if live is set, then we don't finalize the file job
    leave_open: bool=False
    authorization: Optional[str]=None

    @staticmethod
    def from_dict(data: dict) -> 'FinalizeArgs':
        class FinalizeSchema(Schema):
            write_token = fields.Str(required=True)
            replace = fields.Bool(required=False, missing=False)
            force = fields.Bool(required=False, missing=False)
            leave_open = fields.Bool(required=False, missing=False)
            authorization = fields.Str(required=False, missing=None)
        return FinalizeArgs(**FinalizeSchema().load(data))