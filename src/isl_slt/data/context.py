"""Neighbor reconstruction uses verified physical source times, never UID adjacency."""

from collections.abc import Callable

import numpy as np

from isl_slt.config import Context
from isl_slt.data.manifests import Record
from isl_slt.data.pose import PoseSequence


def assemble_context(center: Record, records: list[Record], settings: Context,
                     reader: Callable[[Record], PoseSequence]) -> PoseSequence:
    if not settings.enabled:
        return reader(center)
    if not (center.source_verified and center.timing_verified):
        raise ValueError(f"{center.uid}: context requires verified parent identity, continuity and timestamps")
    source = [r for r in records if r.source_id == center.source_id]
    if any(r.split != center.split for r in source):
        raise ValueError("Context source crosses partitions")
    if any(not r.timing_verified or not r.source_verified for r in source):
        raise ValueError("Every requested context source segment needs verified timing/provenance")
    low = max(0, center.start_seconds - settings.previous_seconds)
    high = center.end_seconds + settings.next_seconds
    if center.source_duration_seconds is not None:
        high = min(high, center.source_duration_seconds)
    selected = sorted([r for r in source if r.end_seconds > low and r.start_seconds < high],
                      key=lambda r: (r.start_seconds, r.uid))
    intervals = [(max(low, r.start_seconds), min(high, r.end_seconds)) for r in selected]
    cursor, gaps = low, []
    for start, end in intervals:
        if start > cursor + 1e-5:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < high - 1e-5:
        gaps.append((cursor, high))
    if gaps and not settings.allow_gaps:
        raise ValueError(f"{center.uid}: requested context unavailable at {gaps}; provide verified coverage/boundaries")
    main = reader(center)
    pieces = []
    for row in selected:
        sequence = main if row.key == center.key else reader(row)
        if set(sequence.names) != set(main.names) or (sequence.width, sequence.height) != (main.width, main.height):
            raise ValueError("Context coordinate schemas or image geometry differ within a source")
        if sequence.names != main.names:
            remap = [sequence.names.index(name) for name in main.names]
            sequence = PoseSequence(sequence.coordinates[:, remap], sequence.valid[:, remap], main.names,
                sequence.times, sequence.segments, sequence.regions, sequence.width, sequence.height, sequence.metadata)
        keep = (sequence.times >= low) & (sequence.times < high)
        for index in np.flatnonzero(keep):
            # Center observations win at duplicated timestamps from overlapping segment files.
            pieces.append((sequence.times[index], row.key != center.key, row.uid, sequence, index))
    pieces.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    unique = []
    for piece in pieces:
        if not unique or abs(piece[0] - unique[-1][0]) > 1e-7:
            unique.append(piece)
    if not unique:
        raise ValueError("Context contains no frames")
    times = np.array([p[0] for p in unique], np.float64)
    segments = np.zeros(len(times), np.int64)
    for index in range(1, len(times)):
        previous, current = unique[index - 1], unique[index]
        new_region = any(previous[0] < end and current[0] >= start for start, end in gaps)
        new_region |= previous[3] is current[3] and (
            previous[3].segments[previous[4]] != current[3].segments[current[4]])
        # A large sampling gap is also a discontinuity, even if annotation bounds abut.
        expected = max(previous[3].metadata.get("sampling_interval_seconds", 1 / max(previous[3].metadata["fps"], 1)),
                       current[3].metadata.get("sampling_interval_seconds", 1 / max(current[3].metadata["fps"], 1)))
        new_region |= current[0] - previous[0] > expected * 1.75
        segments[index] = segments[index - 1] + int(new_region)
    coordinates = np.stack([p[3].coordinates[p[4]] for p in unique])
    valid = np.stack([p[3].valid[p[4]] for p in unique])
    regions = np.where(times < center.start_seconds, -1, np.where(times >= center.end_seconds, 1, 0)).astype(np.int8)
    return PoseSequence(coordinates, valid, main.names, times, segments, regions, main.width, main.height,
                        {**main.metadata, "context_uids": sorted({p[2] for p in unique}), "gaps": gaps,
                         "lookahead_seconds": settings.next_seconds})
