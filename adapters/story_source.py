from __future__ import annotations

from pathlib import Path
from typing import Any

from story_domain import normalize_story_text, take_story_chunk


class StorySourceFinished(EOFError):
    pass


class StorySourceAdapter:
    """Durable, cursor-based access to a pasted story of arbitrary practical length."""

    FILE_NAME = "source.txt"
    READ_BYTES = 256 * 1024

    def persist(self, session_dir: Path, text: str) -> dict[str, Any]:
        target = session_dir / self.FILE_NAME
        staged = target.with_suffix(".txt.tmp")
        staged.write_text(normalize_story_text(text), encoding="utf-8")
        staged.replace(target)
        return {
            "path": self.FILE_NAME,
            "cursor": 0,
            "bytes": target.stat().st_size,
            "complete": False,
        }

    def next_chunk(
        self,
        session_dir: Path,
        source: dict[str, Any],
        *,
        language: str,
        minimum: int,
        maximum: int,
        accepted_minimum: int | None = None,
        accepted_maximum: int | None = None,
    ) -> tuple[str, int]:
        path = (session_dir / str(source.get("path", self.FILE_NAME))).resolve()
        path.relative_to(session_dir.resolve())
        cursor = max(0, int(source.get("cursor", 0)))
        size = path.stat().st_size
        if cursor >= size:
            raise StorySourceFinished

        read_size = self.READ_BYTES
        while True:
            with path.open("rb") as stream:
                stream.seek(cursor)
                raw = stream.read(read_size)
            final = cursor + len(raw) >= size
            text = raw.decode("utf-8", errors="ignore")
            chunk, consumed_chars = take_story_chunk(
                text,
                language,
                minimum,
                maximum,
                final=final,
                accepted_minimum=accepted_minimum,
                accepted_maximum=accepted_maximum,
            )
            if chunk:
                consumed = len(text[:consumed_chars].encode("utf-8"))
                next_cursor = cursor + consumed
                with path.open("rb") as stream:
                    stream.seek(next_cursor)
                    while next_cursor < size and stream.read(1).isspace():
                        next_cursor += 1
                return chunk, next_cursor
            if final:
                tail = text.strip()
                if tail:
                    return tail, size
                raise StorySourceFinished
            read_size = min(size - cursor, read_size * 2)
            if read_size > 4 * 1024 * 1024:
                raise ValueError("My story contains a passage too large to split safely.")

    @staticmethod
    def progress(source: dict[str, Any]) -> float:
        total = max(1, int(source.get("bytes", 0)))
        return min(1.0, max(0.0, int(source.get("cursor", 0)) / total))
