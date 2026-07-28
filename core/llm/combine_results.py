from pathlib import Path
import logging
import shutil

logger = logging.getLogger(__name__)


def combine_jsonl(
    input_paths: list[str | Path],
    output_path: str | Path,
) -> Path:
    """
    Combine multiple JSONL files into a single JSONL file.

    Parameters
    ----------
    input_paths
        JSONL files to combine, in order.
    output_path
        Destination JSONL file.

    Returns
    -------
    Path
        Path to the combined JSONL file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as outfile:
        for path in input_paths:
            with Path(path).open("rb") as infile:
                shutil.copyfileobj(infile, outfile)

    logger.info("Combined %d files into %s", len(input_paths), output_path.name)

    return output_path