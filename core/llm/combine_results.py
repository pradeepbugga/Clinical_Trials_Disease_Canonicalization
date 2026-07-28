from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def combine_jsonl(
    input_files: list[str | Path],
    output_file: str | Path,
) -> Path:
    """
    Combine multiple JSONL files into a single JSONL file.

    Parameters
    ----------
    input_files
        JSONL files to combine, in order.
    output_file
        Destination JSONL file.

    Returns
    -------
    Path
        Path to the combined JSONL file.
    """

    output_file = Path(output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as outfile:

        for file in input_files:

            with Path(file).open("r", encoding="utf-8") as infile:
                outfile.write(infile.read())

    logger.info("Combined %d files into %s", len(input_files), output_file.name)

    return output_file
