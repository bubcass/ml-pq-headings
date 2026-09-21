#!/usr/bin/env python3
"""Extract a daily PQ input dataset from an Oireachtas question-paper DOCX.

The output keeps the established standalone-test columns, including the paper
paragraph locations and PQ reference number needed to reconcile with later
published web data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from preprocess import clean_pq


WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": WORD_NAMESPACE}
DEPARTMENT_HEADER_RE = re.compile(r"^Chun .+?:\s*To (?:the )?(.+)$")
QUESTION_RE = re.compile(r"^(\d+)\.\s+(.+)$")
REFERENCE_RE = re.compile(r"^PQ(\d{1,6}/\d{2})\s+(.+?)$")
ENGLISH_DATE_RE = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
    r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),\s+(\d{4})"
)
ENGLISH_DATE_ANYWHERE_RE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),\s+(\d{4})", re.IGNORECASE)
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
OUTPUT_COLUMNS = [
    "question",
    "department",
    "question_clean",
    "date",
    "answer_type",
    "question_number",
    "refNo",
    "deputy",
    "deputy_source",
    "language",
    "raw_question",
    "department_paper",
    "source_question_paragraph",
    "source_pq_paragraph",
]


def normalise_space(value: str) -> str:
    return " ".join(value.replace("\u00a0", " ").split())


def read_docx_paragraphs(path: Path) -> list[str]:
    """Read visible main-document paragraph text without an external DOCX library."""
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
    except FileNotFoundError:
        raise SystemExit(f"Question paper not found: {path}")
    except (KeyError, zipfile.BadZipFile) as error:
        raise SystemExit(f"Not a readable DOCX question paper: {path} ({error})")

    root = ET.fromstring(xml)
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:body/w:p", NS):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", NS))
        paragraphs.append(normalise_space(text))
    return paragraphs


def paper_date(paragraphs: Iterable[str], override: str | None) -> str:
    if override:
        try:
            return date.fromisoformat(override).isoformat()
        except ValueError:
            raise SystemExit(f"--date must be YYYY-MM-DD, received {override!r}")

    for paragraph in paragraphs:
        match = ENGLISH_DATE_RE.search(paragraph)
        if match:
            day, month_name, year = match.groups()
            month = MONTHS.get(month_name.lower())
            if month:
                return date(int(year), month, int(day)).isoformat()
    raise SystemExit("Could not find the paper date. Supply it with --date YYYY-MM-DD.")


def date_in_text(value: str) -> str | None:
    match = ENGLISH_DATE_ANYWHERE_RE.search(value)
    if not match:
        return None
    day, month_name, year = match.groups()
    month = MONTHS.get(month_name.lower())
    if not month:
        return None
    return date(int(year), month, int(day)).isoformat()


def next_non_empty(paragraphs: list[str], start: int) -> tuple[int, str] | None:
    for index in range(start, len(paragraphs)):
        if paragraphs[index]:
            return index, paragraphs[index]
    return None


def display_deputy(deputy_source: str) -> str:
    particles = {"the", "de", "del", "der", "la", "le", "van", "von"}
    words = deputy_source.lower().split()
    def format_word(word: str) -> str:
        formatted = word.capitalize()
        return re.sub(r"(?<=')[a-z]", lambda match: match.group(0).upper(), formatted)

    return " ".join(word if index and word in particles else format_word(word) for index, word in enumerate(words))


def question_display(question_number: str, raw_question: str, deputy: str, ref_no: str) -> str:
    if raw_question.lower().startswith("to ask "):
        text = f"{question_number}. Deputy {deputy} asked {raw_question[7:]}"
    else:
        text = f"{question_number}. Deputy {deputy} {raw_question}"
    return f"{text} [{ref_no}]"


def load_aliases(path: Path) -> dict[str, str]:
    try:
        aliases = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Department alias file not found: {path}")
    if not isinstance(aliases, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()):
        raise SystemExit(f"Department alias file must contain a string-to-string JSON object: {path}")
    return {normalise_space(key): normalise_space(value) for key, value in aliases.items()}


def extract_rows(paragraphs: list[str], aliases: dict[str, str], source_date: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    stats = {"question_paragraphs": 0, "accepted": 0, "unmatched_question_paragraphs": 0}
    department_paper = ""
    answer_type = "written"
    answer_date = source_date

    for index, paragraph in enumerate(paragraphs):
        header = DEPARTMENT_HEADER_RE.match(paragraph)
        if header:
            department_paper = normalise_space(header.group(1))
            continue

        upper_paragraph = paragraph.upper()
        if upper_paragraph.startswith("QUESTIONS") and "ANSWER" in upper_paragraph:
            if "ORAL ANSWER" in upper_paragraph:
                answer_type = "oral"
            section_date = date_in_text(paragraph)
            if section_date:
                answer_date = section_date
            continue

        question_match = QUESTION_RE.match(paragraph)
        if not question_match:
            continue
        question_number, raw_question = question_match.groups()
        raw_question = normalise_space(raw_question)
        if not raw_question.lower().startswith(("to ask ", "chun a fhiafraí ")):
            continue

        stats["question_paragraphs"] += 1
        pq_line = next_non_empty(paragraphs, index + 1)
        if pq_line is None:
            stats["unmatched_question_paragraphs"] += 1
            continue
        pq_index, pq_text = pq_line
        reference = REFERENCE_RE.match(pq_text)
        if not reference:
            stats["unmatched_question_paragraphs"] += 1
            continue
        if not department_paper:
            raise SystemExit(f"No department header before question paragraph {index}.")
        if department_paper not in aliases:
            raise SystemExit(
                f"No department alias for {department_paper!r}. Add it to the alias JSON file before continuing."
            )

        ref_no, deputy_source = reference.groups()
        deputy_source = normalise_space(deputy_source)
        deputy = display_deputy(deputy_source)
        language = "en" if raw_question.lower().startswith("to ask ") else "ga"
        question = question_display(question_number, raw_question, deputy, ref_no)
        rows.append(
            {
                "question": question,
                "department": aliases[department_paper],
                "question_clean": clean_pq(question),
                "date": answer_date,
                "answer_type": answer_type,
                "question_number": question_number,
                "refNo": ref_no,
                "deputy": deputy,
                "deputy_source": deputy_source,
                "language": language,
                "raw_question": raw_question,
                "department_paper": department_paper,
                "source_question_paragraph": str(index),
                "source_pq_paragraph": str(pq_index),
            }
        )
        stats["accepted"] += 1

    return rows, stats


def validate_rows(rows: list[dict[str, str]], stats: dict[str, int]) -> None:
    if not rows:
        raise SystemExit("No questions extracted from the paper.")
    missing = [column for column in OUTPUT_COLUMNS if any(not row[column] for row in rows)]
    if missing:
        raise SystemExit(f"Extracted rows contain blank required values: {missing}")
    if stats["unmatched_question_paragraphs"]:
        raise SystemExit(
            f"{stats['unmatched_question_paragraphs']} question paragraphs lacked a following PQ reference line. "
            "The paper format may have changed."
        )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Extract a daily ML input CSV from an Oireachtas question-paper DOCX.")
    parser.add_argument("--input", type=Path, required=True, help="Question-paper DOCX")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path; defaults to outputs/word_ml_inputs/<paper-date>_questions_ml_input.csv",
    )
    parser.add_argument("--date", help="Paper date override in YYYY-MM-DD format")
    parser.add_argument(
        "--department-aliases",
        type=Path,
        default=root / "config" / "word_paper_department_aliases.json",
        help="JSON mapping from paper department names to model department labels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    paragraphs = read_docx_paragraphs(args.input)
    source_date = paper_date(paragraphs, args.date)
    aliases = load_aliases(args.department_aliases)
    rows, stats = extract_rows(paragraphs, aliases, source_date)
    validate_rows(rows, stats)

    output_path = args.output or root / "outputs" / "word_ml_inputs" / f"{source_date}_questions_ml_input.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=output_path.parent, delete=False, prefix=f".{output_path.name}."
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, output_path)

    unique_refs = len({row["refNo"] for row in rows})
    print(f"Paper date: {source_date}")
    print(f"Extracted rows: {len(rows):,}")
    print(f"Unique PQ references: {unique_refs:,}")
    print(f"Answer types: written={sum(r['answer_type'] == 'written' for r in rows):,}, oral={sum(r['answer_type'] == 'oral' for r in rows):,}")
    print(f"Languages: English={sum(r['language'] == 'en' for r in rows):,}, Irish={sum(r['language'] == 'ga' for r in rows):,}")
    print(f"Extraction audit: {json.dumps(stats, sort_keys=True)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
