"""Pure local transformation used by the fixture; no external side effect."""


def normalize(record: dict[str, object]) -> dict[str, object]:
    allowed = ("record_id", "status", "updated_at")
    return {key: record[key] for key in allowed if key in record}
