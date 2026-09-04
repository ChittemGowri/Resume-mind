"""Safe PDF resume text extraction."""
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Union

import pdfplumber


class ResumeParseError(ValueError):
    """Raised when a supplied document cannot be used as a text resume."""


def extract_resume_text(source: Union[str, Path, bytes, BinaryIO]) -> str:
    """Extract selectable text from a PDF source with friendly, actionable errors."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() != ".pdf":
            raise ResumeParseError("Please upload a PDF resume.")
        if not path.exists():
            raise ResumeParseError("The selected resume file could not be found.")
        stream = str(path)
    elif isinstance(source, bytes):
        if not source:
            raise ResumeParseError("The uploaded file is empty.")
        stream = BytesIO(source)
    else:
        stream = source

    try:
        with pdfplumber.open(stream) as pdf:
            if not pdf.pages:
                raise ResumeParseError("This PDF has no pages.")
            text = "\n".join((page.extract_text() or "") for page in pdf.pages).strip()
    except ResumeParseError:
        raise
    except Exception as exc:
        raise ResumeParseError("We couldn't read this PDF. Please upload a valid, non-password-protected PDF.") from exc

    if len(text) < 25:
        raise ResumeParseError(
            "No readable resume text was found. This may be a scanned PDF; please use a text-based PDF."
        )
    return text

