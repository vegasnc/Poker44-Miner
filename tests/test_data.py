from __future__ import annotations

from src.data.loader import iter_examples_from_records


def test_iter_examples_pairs_chunks_and_labels(sample_chunk_group):
    records = [
        {
            "chunkId": "abc",
            "chunkHash": "hash",
            "sourceDate": "2026-06-24",
            "releaseVersion": "v1",
            "schemaVersion": "test",
            "chunks": [sample_chunk_group, sample_chunk_group],
            "groundTruth": [0, 1],
        }
    ]

    examples = iter_examples_from_records(records)

    assert len(examples) == 2
    assert examples[0].label == 0
    assert examples[1].label == 1
    assert examples[0].chunk == sample_chunk_group
