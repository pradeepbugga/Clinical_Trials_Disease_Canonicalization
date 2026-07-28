CANONICALIZATION_SCHEMA = {
    "type": "object",
    "properties": {
        "common_name": {
            "type": "string",
        },
        "technical_name": {
            "type": "string",
        },
        "abbreviations": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": [
        "common_name",
        "technical_name",
        "abbreviations",
    ],
    "additionalProperties": False,
}
