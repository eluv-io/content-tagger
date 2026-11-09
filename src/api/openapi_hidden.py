"""Support for request fields which are accepted by the API but left out of the OpenAPI docs.

Mark a marshmallow field with ``metadata={DOC_HIDDEN: True}``; it still validates and loads
as usual, but the converter below skips it when rendering the spec.
"""

from apispec.ext.marshmallow import MarshmallowPlugin
from apispec.ext.marshmallow.openapi import OpenAPIConverter

DOC_HIDDEN = "doc_hidden"


class HiddenFieldsConverter(OpenAPIConverter):
    def fields2jsonschema(self, fields, *, partial=None):
        visible = {name: f for name, f in fields.items() if not f.metadata.get(DOC_HIDDEN)}
        return super().fields2jsonschema(visible, partial=partial)


class HiddenFieldsMarshmallowPlugin(MarshmallowPlugin):
    Converter = HiddenFieldsConverter
