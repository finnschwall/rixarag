from dataclasses import dataclass
from enum import Enum
import fitz  # PyMuPDF
import re
from typing import List, Dict, Optional, Tuple
import numpy as np

class Granularity(Enum):
    SECTION = "section"
    PARAGRAPH = "paragraph"

total_total = 0
total_matched = 0

def reset_total():
    global total_total
    global total_matched
    total_total = 0
    total_matched = 0

@dataclass
class Chunk:
    """Represents a chunk of text with metadata"""
    content: str
    start_pos: int
    end_pos: int
    separator_type: str
    granularity: Granularity
    size: int


class PDFLatexMatcher:
    def __init__(self, pdf_path: str, chunks: List[Chunk]):
        self.pdf_doc = fitz.open(pdf_path)
        self.chunks = chunks
        self.page_assignments: Dict[int, Optional[int]] = {i: None for i in range(len(chunks))}

    def extract_section_title(self, chunk: Chunk) -> Optional[str]:
        """Extract section title from chunk if it exists."""
        section_patterns = [
            r'\\section\{([^}]+)\}',
            r'\\subsection\{([^}]+)\}',
            r'\\subsubsection\{([^}]+)\}'
        ]

        for pattern in section_patterns:
            match = re.search(pattern, chunk.content)
            if match:
                return match.group(1).strip()
        return None

    def find_text_in_pdf(self, text: str) -> List[Tuple[int, float]]:
        """Find all occurrences of text in PDF, returns list of (page_num, confidence)."""
        matches = []
        for page_num in range(len(self.pdf_doc)):
            page = self.pdf_doc[page_num]
            # Search for text in page
            instances = page.search_for(text)
            if instances:
                # Calculate confidence based on exact match and uniqueness
                confidence = 1.0 if len(instances) == 1 else 0.5
                matches.append((page_num, confidence))
        return matches

    def assign_unique_matches(self):
        """First pass: Assign pages for chunks with unique section matches."""
        for chunk_idx, chunk in enumerate(self.chunks):
            title = self.extract_section_title(chunk)
            if not title:
                continue

            matches = self.find_text_in_pdf(title)
            if len(matches) == 1:
                self.page_assignments[chunk_idx] = matches[0][0]

    def resolve_ambiguous_matches(self):
        """Second pass: Resolve ambiguous matches using known boundaries."""

        def find_nearest_assigned_chunks(idx: int) -> Tuple[Optional[int], Optional[int]]:
            """Find nearest assigned chunks before and after given index."""
            prev_assigned = None
            next_assigned = None

            # Look backwards
            for i in range(idx - 1, -1, -1):
                if self.page_assignments[i] is not None:
                    prev_assigned = i
                    break

            # Look forwards
            for i in range(idx + 1, len(self.chunks)):
                if self.page_assignments[i] is not None:
                    next_assigned = i
                    break

            return prev_assigned, next_assigned

        for chunk_idx, chunk in enumerate(self.chunks):
            if self.page_assignments[chunk_idx] is not None:
                continue

            title = self.extract_section_title(chunk)
            if not title:
                continue

            prev_idx, next_idx = find_nearest_assigned_chunks(chunk_idx)
            if prev_idx is None or next_idx is None:
                continue

            prev_page = self.page_assignments[prev_idx]
            next_page = self.page_assignments[next_idx]

            # Look for matches only within the bounded range
            matches = self.find_text_in_pdf(title)
            valid_matches = [
                (page, conf) for page, conf in matches
                if prev_page <= page <= next_page
            ]

            if len(valid_matches) == 1:
                self.page_assignments[chunk_idx] = valid_matches[0][0]

    def interpolate_missing_pages(self):
        """Final pass: Interpolate pages for unassigned chunks."""
        return_chunks = []

        for chunk_idx in range(len(self.chunks)):
            if self.page_assignments[chunk_idx] is not None:
                try:
                    page_label = self.pdf_doc[self.page_assignments[chunk_idx]].get_label()
                    if page_label is None or page_label == "":
                        page_label = str(self.page_assignments[chunk_idx]+1)
                    return_chunks.append(page_label)
                except:
                    return_chunks.append("Unexpected error")
                continue

            prev_idx, next_idx = None, None
            prev_page, next_page = None, None

            # Find previous assigned chunk
            for i in range(chunk_idx - 1, -1, -1):
                if self.page_assignments[i] is not None:
                    prev_idx = i
                    prev_page = self.page_assignments[i]
                    break

            # Find next assigned chunk
            for i in range(chunk_idx + 1, len(self.chunks)):
                if self.page_assignments[i] is not None:
                    next_idx = i
                    next_page = self.page_assignments[i]
                    break
            try:
                if next_page is not None:
                    next_page_label = self.pdf_doc[next_page].get_label()
                    if next_page_label is None or next_page_label == "":
                        next_page_label = str(next_page+1)
                if prev_page is not None:
                    prev_page_label = self.pdf_doc[prev_page].get_label()
                    if prev_page_label is None or prev_page_label == "":
                        prev_page_label = str(prev_page+1)
            except:
                return_chunks.append("Unexpected error")
            if prev_page is not None and next_page is not None:
                self.page_assignments[chunk_idx] = prev_page
                return_chunks.append(next_page_label if next_page == prev_page else f"{prev_page_label}-{next_page_label}")
            elif prev_page is not None:
                return_chunks.append(prev_page_label)
                self.page_assignments[chunk_idx] = prev_page
            elif next_page is not None:
                return_chunks.append(next_page_label)
                self.page_assignments[chunk_idx] = next_page
            else:
                return_chunks.append(str("Error"))
        return return_chunks

    def find_matchable_sequence(self, chunk: Chunk) -> Optional[str]:
        """Find a LaTeX-free sequence in the chunk that could be matched in the PDF."""

        def has_latex_commands(text: str) -> bool:
            """Check if text contains LaTeX commands or special characters."""
            latex_patterns = [
                r'\\[a-zA-Z]+\{',  # Commands with braces
                r'\\[a-zA-Z]+\[',  # Commands with square brackets
                r'\$.*?\$',  # Inline math
                r'\\[a-zA-Z]+',  # Simple commands
                r'\{.*?\}',  # Braced content
                r'\[.*?\]'  # Bracketed content
            ]
            return any(re.search(pattern, text) for pattern in latex_patterns)

        def split_into_sentences(text: str) -> List[str]:
            """Split text into sentences, handling common abbreviations."""
            # Add spaces after periods that aren't part of common abbreviations
            text = re.sub(r'(?<!Mr)(?<!Dr)(?<!Mrs)(?<!Ms)(?<!vs)(?<!etc)(?<!i.e)(?<!e.g)\.\s+', '.|', text)
            return [s.strip() for s in text.split('|') if s.strip()]

        # Try to find a clean sentence at the start of the chunk
        first_para = chunk.content.split('\n\n')[0] if '\n\n' in chunk.content else chunk.content
        sentences = split_into_sentences(first_para)

        for sentence in sentences:
            # Skip very short sentences
            if len(sentence) < 10:
                continue

            # Skip sentences with LaTeX
            if has_latex_commands(sentence):
                continue

            # Limit length to reduce wrapping issues (around 100 chars)
            if len(sentence) > 100:
                continue

            # Make sure the sentence has actual content
            if not re.search(r'[a-zA-Z]{3,}', sentence):
                continue

            return sentence.strip()

        # If no suitable sentence found at start, try end of chunk
        if '\n\n' in chunk.content:
            last_para = chunk.content.split('\n\n')[-1]
            sentences = split_into_sentences(last_para)

            for sentence in sentences:
                if (len(sentence) >= 10 and
                        len(sentence) <= 100 and
                        not has_latex_commands(sentence) and
                        re.search(r'[a-zA-Z]{3,}', sentence)):
                    return sentence.strip()

        return None

    def assign_content_matches(self):
        """Match chunks to pages based on unique content sequences."""
        for chunk_idx, chunk in enumerate(self.chunks):
            # Skip if already assigned
            if self.page_assignments[chunk_idx] is not None:
                continue

            matchable_sequence = self.find_matchable_sequence(chunk)
            if not matchable_sequence:
                continue

            matches = self.find_text_in_pdf(matchable_sequence)

            # If we have exactly one match, assign it
            if len(matches) == 1:
                self.page_assignments[chunk_idx] = matches[0][0]
            # If we have multiple matches but know the boundaries, try to narrow it down
            elif len(matches) > 1:
                prev_idx, next_idx = None, None
                prev_page = next_page = None

                # Find previous assigned chunk
                for i in range(chunk_idx - 1, -1, -1):
                    if self.page_assignments[i] is not None:
                        prev_idx = i
                        prev_page = self.page_assignments[i]
                        break

                # Find next assigned chunk
                for i in range(chunk_idx + 1, len(self.chunks)):
                    if self.page_assignments[i] is not None:
                        next_idx = i
                        next_page = self.page_assignments[i]
                        break

                # Filter matches within known boundaries
                valid_matches = []
                for page_num, confidence in matches:
                    if prev_page is not None and page_num < prev_page:
                        continue
                    if next_page is not None and page_num > next_page:
                        continue
                    valid_matches.append((page_num, confidence))

                if len(valid_matches) == 1:
                    self.page_assignments[chunk_idx] = valid_matches[0][0]

    def match_section_to_toc(self):
        # TODO in case of no obtainable TOC it could be possible to still try to match based on fontsize, font etc.
        # Usually sections will always have a different font. But for that an analysis on the font makeup of the document would be needed
        toc = self.pdf_doc.get_toc()
        if not toc:
            return
        for chunk_idx, chunk in enumerate(self.chunks):
            # Skip if already assigned
            if self.page_assignments[chunk_idx] is not None:
                continue
            matchable_sequence = self.extract_section_title(chunk)

            if not matchable_sequence:
                continue

            page_assignment = None
            for entry in toc:
                if matchable_sequence in entry[1]:
                    if page_assignment is None:
                        page_assignment = entry[2]
                    else:
                        page_assignment = "Invalid"
            if page_assignment is not None and page_assignment != "Invalid":
                self.page_assignments[chunk_idx] = page_assignment

    def get_ratio_assigned(self):
        return np.sum(np.array([True if x is not None else False for x in self.page_assignments.values()])) / len(self.page_assignments)

    def match_chunks_to_pages(self) -> Dict[int, int]:
        global total_total, total_matched
        """Execute the full matching process."""
        # print(len(self.pdf_doc))
        # print(int(self.pdf_doc[-1]))
        # print()

        self.match_section_to_toc()
        # print(f"Iteration 1 (ToC based): {self.get_ratio_assigned():.2f} % chunks were uniquely assigned pages")

        # Assign unique matches by looking for (sub)(sub)section titles that can be *uniquely* matched
        self.assign_unique_matches()
        # print(f"Iteration 2 (section based): {self.get_ratio_assigned():.2f} % chunks were uniquely assigned pages")

        # try to match based on latex-free sentences (again uniquely)
        self.assign_content_matches()
        # print(f"Iteration 3 (content based): {self.get_ratio_assigned():.2f} % chunks were uniquely assigned pages")

        # Attempt to redo step 1 but now only searching within known boundaries established by previous methods
        self.resolve_ambiguous_matches()
        # print(f"Iteration 4 (section based with constraints): {self.get_ratio_assigned():.2f} % chunks were uniquely assigned pages")
        # TODO fix major issue when chunks are created using multiple .tex files. The page at the boundaries will be wrong which will cause the interpolation to fail
        # To fix this whole thing needs to be integrated into the latex chunker to be file aware

        matched = np.sum(np.array([True if x is not None else False for x in self.page_assignments.values()]))

        # Assign ranges to remaining unassigned chunks
        return_list = self.interpolate_missing_pages()
        total = len(self.chunks)

        total_total += total
        total_matched += matched

        return return_list

    def close(self):
        """Clean up PDF document."""
        self.pdf_doc.close()
