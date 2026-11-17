from collections import defaultdict


def prepare_reliable_sources(chunks: list[dict], current_versions: dict[str, str] | None = None) -> tuple[list[dict], set[str]]:
    """Drop known stale sources and mark explicit source-group conflicts."""
    current = []
    for chunk in chunks:
        expected = (current_versions or {}).get(str(chunk.get("file", "")))
        if chunk.get("obsolete") or (expected is not None and chunk.get("source_version") != expected):
            continue
        current.append(dict(chunk))

    groups = defaultdict(list)
    for chunk in current:
        if chunk.get("source_group"):
            groups[chunk["source_group"]].append(chunk)
    contradictions = {
        group for group, sources in groups.items()
        if len({source.get("chunk_version", source.get("content")) for source in sources}) > 1
    }
    for chunk in current:
        if chunk.get("source_group") in contradictions:
            chunk["reliability"] = "contradictory"
    return current, contradictions
