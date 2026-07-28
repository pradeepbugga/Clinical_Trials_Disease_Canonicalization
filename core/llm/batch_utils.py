from pathlib import Path
from collections.abc import Callable, Iterable
import json
from typing import Any


def generate_batch_jsonl(
    records: Iterable,
    output_path: str | Path,
    build_request: Callable,
) -> dict[str, Any]:
    """
    Generate an OpenAI Batch API JSONL file.

    Parameters
    ----------
    records
        Iterable of records (e.g. Trial objects).
    output_path
        Path to the JSONL file.
    build_request
        Function that converts a record into a Batch API request.

    Returns
    -------
    dict
        Summary statistics.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    total_written = 0

    with output_path.open("w", encoding="utf-8") as f:

        for record in records:

            total_records += 1

            task = build_request(record)

            if task is None:
                continue

            f.write(json.dumps(task) + "\n")
            total_written += 1

    return {
        "total_records": total_records,
        "total_written": total_written,
    }


def split_jsonl(
    input_path: str | Path, output_dir: str | Path, max_requests: int = 25000
) -> list[Path]:
    """
    Split a JSONL file into multiple smaller JSONL files.

    Parameters
    ----------
    input_path
        Path to the input JSONL file.
    output_dir
        Directory where split files will be written.
    max_requests
        Maximum number of records per file. Defaults to the
        current OpenAI Batch API limit (25,000).

    Returns
    -------
    list[Path]
        Paths to the generated JSONL files.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem

    output_paths = []

    part = 1
    line_count = 0

    output_path = output_dir / f"{stem}_part_{part}.jsonl"
    output_paths.append(output_path)
    out_file = output_path.open("w")

    with input_path.open("r", encoding="utf-8") as in_file:

        for line in in_file:

            if line_count == max_requests:
                out_file.close()

                part += 1
                line_count = 0

                output_path = output_dir / f"{stem}_part_{part}.jsonl"
                output_paths.append(output_path)
                out_file = output_path.open("w")

            out_file.write(line)
            line_count += 1

    out_file.close()

    return output_paths


def combine_jsonl(
    input_paths: list[str | Path],
    output_path: str | Path,
    unique_key: str | None = None,
) -> Path:
    """
    Combine multiple JSONL files into a single JSONL file.

    Parameters
    ----------
    input_paths
        Paths to the input JSONL files.
    output_path
        Path to the combined JSONL file.
    unique_key
        Optional JSON field used to remove duplicate records.
        If None, all records are preserved.

    Returns
    -------
    Path
        Path to the combined JSONL file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()

    total_records = 0
    unique_records = 0
    duplicates = 0

    with output_path.open("w", encoding="utf-8") as out_file:

        for input_path in input_paths:

            input_path = Path(input_path)

            with input_path.open("r", encoding="utf-8") as in_file:

                for line in in_file:

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        print(
                            f"Skipping invalid JSON in {input_path}: " f"{line.strip()}"
                        )
                        continue

                    total_records += 1

                    # No deduplication requested
                    if unique_key is None:
                        out_file.write(line)
                        unique_records += 1
                        continue

                    key = record.get(unique_key)

                    # Missing key → keep record
                    if key is None:
                        out_file.write(line)
                        unique_records += 1
                        continue

                    if key in seen:
                        duplicates += 1
                        continue

                    seen.add(key)
                    out_file.write(line)
                    unique_records += 1

    print(
        f"Combined {total_records} records into "
        f"{unique_records} records "
        f"({duplicates} duplicates removed)."
    )

    return output_path
