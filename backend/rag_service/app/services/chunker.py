"""Document chunking utilities."""

from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk of text from a document."""

    id: str
    text: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split text into overlapping chunks.

    Args:
        text: The text to chunk.
        document_id: Identifier for the source document.
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Overlap between chunks in characters.
        metadata: Additional metadata to attach to each chunk.

    Returns:
        List of Chunk objects.
    """
    if not text:
        return []

    metadata = metadata or {}
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        # Find end position
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending punctuation
            for punct in [". ", ".\n", "? ", "?\n", "! ", "!\n"]:
                last_punct = text.rfind(punct, start, end)
                if last_punct > start + chunk_size // 2:
                    end = last_punct + 1
                    break

        # Ensure we don't go past the text
        end = min(end, len(text))

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata=metadata.copy(),
                )
            )
            chunk_index += 1

        # Move start position with overlap
        start = end - chunk_overlap
        if start <= chunks[-1].start_char if chunks else 0:
            start = end

    return chunks


def chunk_by_section(
    text: str,
    document_id: str,
    section_pattern: str = r"\n(?=Article \d+)",
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split text by section headers (e.g., articles).

    Args:
        text: The text to chunk.
        document_id: Identifier for the source document.
        section_pattern: Regex pattern for section boundaries.
        metadata: Additional metadata to attach to each chunk.

    Returns:
        List of Chunk objects.
    """
    import re

    if not text:
        return []

    metadata = metadata or {}
    chunks = []

    # Split by pattern
    parts = re.split(section_pattern, text)
    current_pos = 0

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        chunk_id = f"{document_id}_section_{i}"
        start_char = text.find(part, current_pos)
        end_char = start_char + len(part)

        # Try to extract section title
        section_metadata = metadata.copy()
        first_line = part.split("\n")[0]
        if first_line:
            section_metadata["section_title"] = first_line[:100]

        chunks.append(
            Chunk(
                id=chunk_id,
                text=part,
                document_id=document_id,
                chunk_index=i,
                start_char=start_char,
                end_char=end_char,
                metadata=section_metadata,
            )
        )
        current_pos = end_char

    return chunks
