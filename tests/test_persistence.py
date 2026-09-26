import json

import pytest

from research_rag.ingestion.persistence import (save_chunk_records, load_chunk_records)


def test_save_chunk_records_overwrite(tmp_path):
    chunk_records = [
        {
            "document_id": "paper1.pdf",
            "chunk_id": "chunk-1",
            "text": "First chunk",
        },
        {
            "document_id": "paper1.pdf",
            "chunk_id": "chunk-2",
            "text": "Second chunk",
        },
    ]

    output_path = tmp_path / "processed" / "chunk_records.jsonl"

    save_chunk_records(
        chunk_records=chunk_records,
        output_path=output_path,
        mode="overwrite",
    )

    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as f:
        saved_records = [json.loads(line) for line in f if line.strip()]

    assert saved_records == chunk_records


def test_save_chunk_records_creates_parent_directory(tmp_path):
    output_path = (
        tmp_path
        / "some"
        / "nested"
        / "directory"
        / "chunk_records.jsonl"
    )

    chunk_records = [
        {
            "document_id": "paper1.pdf",
            "chunk_id": "chunk-1",
            "text": "Test chunk",
        }
    ]

    assert not output_path.parent.exists()

    save_chunk_records(
        chunk_records=chunk_records,
        output_path=output_path,
        mode="overwrite",
    )

    assert output_path.parent.exists()
    assert output_path.exists()


def test_save_chunk_records_append(tmp_path):
    output_path = tmp_path / "chunk_records.jsonl"

    first_records = [
        {
            "document_id": "paper1.pdf",
            "chunk_id": "chunk-1",
            "text": "First chunk",
        }
    ]

    second_records = [
        {
            "document_id": "paper2.pdf",
            "chunk_id": "chunk-2",
            "text": "Second chunk",
        }
    ]

    save_chunk_records(
        chunk_records=first_records,
        output_path=output_path,
        mode="overwrite",
    )

    save_chunk_records(
        chunk_records=second_records,
        output_path=output_path,
        mode="append",
    )

    with open(output_path, "r", encoding="utf-8") as f:
        saved_records = [json.loads(line) for line in f if line.strip()]

    assert saved_records == first_records + second_records


def test_save_chunk_records_overwrite_replaces_existing_records(tmp_path):
    output_path = tmp_path / "chunk_records.jsonl"

    old_records = [
        {
            "document_id": "old.pdf",
            "chunk_id": "old-chunk",
            "text": "Old data",
        }
    ]

    new_records = [
        {
            "document_id": "new.pdf",
            "chunk_id": "new-chunk",
            "text": "New data",
        }
    ]

    save_chunk_records(
        chunk_records=old_records,
        output_path=output_path,
        mode="overwrite",
    )

    save_chunk_records(
        chunk_records=new_records,
        output_path=output_path,
        mode="overwrite",
    )

    with open(output_path, "r", encoding="utf-8") as f:
        saved_records = [json.loads(line) for line in f if line.strip()]

    assert saved_records == new_records


def test_save_chunk_records_invalid_mode(tmp_path):
    output_path = tmp_path / "chunk_records.jsonl"

    with pytest.raises(ValueError):
        save_chunk_records(
            chunk_records=[],
            output_path=output_path,
            mode="invalid",
        )
        
def test_load_chunk_records(tmp_path):
    output_path = tmp_path / "chunk_records.jsonl"

    chunk_records = [
        {
            "document_id": "paper1.pdf",
            "chunk_id": "chunk-1",
            "text": "First chunk",
        },
        {
            "document_id": "paper2.pdf",
            "chunk_id": "chunk-2",
            "text": "Second chunk",
        },
    ]

    save_chunk_records(
        chunk_records=chunk_records,
        output_path=output_path,
        mode="overwrite",
    )

    valid_records, load_errors = load_chunk_records(output_path)

    assert valid_records == chunk_records
    assert load_errors == []


def test_load_chunk_records_skips_blank_lines(tmp_path):
    output_path = tmp_path / "chunk_records.jsonl"

    output_path.write_text(
        '{"document_id": "paper1.pdf", "chunk_id": "chunk-1"}\n'
        '\n'
        '   \n'
        '{"document_id": "paper2.pdf", "chunk_id": "chunk-2"}\n',
        encoding="utf-8",
    )

    valid_records, load_errors = load_chunk_records(output_path)

    assert len(valid_records) == 2
    assert load_errors == []


def test_load_chunk_records_reports_malformed_lines(tmp_path):
    output_path = tmp_path / "chunk_records.jsonl"

    output_path.write_text(
        '{"document_id": "paper1.pdf", "chunk_id": "chunk-1"}\n'
        '{"document_id": "broken.pdf", BAD JSON}\n'
        '{"document_id": "paper2.pdf", "chunk_id": "chunk-2"}\n',
        encoding="utf-8",
    )

    valid_records, load_errors = load_chunk_records(output_path)

    assert len(valid_records) == 2
    assert len(load_errors) == 1

    assert load_errors[0]["line_number"] == 2
    assert "BAD JSON" in load_errors[0]["raw_line"]
    assert load_errors[0]["error_message"]


def test_load_chunk_records_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(FileNotFoundError):
        load_chunk_records(missing_path)