import re
from typing import List, Dict, Union, Literal
from dataclasses import dataclass
from enum import Enum
import numpy as np

def generate_id(text):
    # hopefully collision free
    return str(int(abs(hash(text))%2e7))


class Granularity(str, Enum):
    COARSE = "coarse"  # Major structural elements (chapter, section, subsection)
    MEDIUM = "medium"  # Minor structural elements (paragraph, subparagraph)
    FINE = "fine"  # List items, environments, definitions
    FINEST = "finest"  # Paragraphs (double newlines)


markdown_separator_patterns = separator_patterns = {
    Granularity.COARSE: {
        "h1": r'^#\s+[^\n]+$',  # # Header
        "h1_alternate": r'^[^\n]+\n=+$',  # Header
        # ======
        "document_divider": r'^---\s*$',  # Horizontal rules/document dividers
    },

    Granularity.MEDIUM: {
        "h2": r'^##\s+[^\n]+$',  # ## Header
        "h2_alternate": r'^[^\n]+\n-+$',  # Header
        # ------
        "h3": r'^###\s+[^\n]+$',  # ### Header
    },

    Granularity.FINE: {
        "h4": r'^####\s+[^\n]+$',  # #### Header
        "h5": r'^#####\s+[^\n]+$',  # ##### Header
        "h6": r'^######\s+[^\n]+$',  # ###### Header
        "blockquote": r'^>\s+[^\n]+$',  # > Blockquote
        "code_block": r'^```[^`]*```$',  # ```code blocks```
        "list_item": r'^\s*[-\*\+]\s+',  # - List items
        "numbered_list": r'^\s*\d+\.\s+',  # 1. Numbered items
        "table": r'^\|[^\n]+\|$',  # |table|rows|
    },

    Granularity.FINEST: {
        "double_newline": r'\n\s*\n\s*',  # Paragraph breaks
        "thematic_break": r'^[\*\-_]{3,}\s*$',  # ***, ---, ___
        "emphasis": r'[\*_]{1,2}[^\*_]+[\*_]{1,2}',  # *emphasis* or **strong**
        "link": r'\[[^\]]+\]\([^\)]+\)',  # [links](url)
    }
}


@dataclass
class Chunk:
    """Represents a chunk of text with metadata"""
    content: str
    start_pos: int
    end_pos: int
    separator_type: str
    granularity: Granularity
    size: int

    def __post_init__(self):
        self.content = self.content.strip()


class RegexChunker:
    def __init__(
            self,
            min_chunk_length: int = 200,
            max_chunk_length: int = 2000,
            custom_separators: Dict[str, str] = None
    ):
        """
        Initialize the LaTeX chunker with configurable length constraints.

        :param min_chunk_length: Minimum length for a chunk before merging
        :param max_chunk_length: Maximum length for any chunk
        :param custom_separators: Optional dictionary of custom separator patterns
        """
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.custom_separators = custom_separators or {}

        # Initialize separator patterns for different granularities
        self._initialize_separator_patterns()

    def _make_brace_pattern(self):
        """Creates pattern for matching content in curly braces, handling nested braces"""
        return r'\{(?:[^{}]|(?:\{[^{}]*\})|(?:\{(?:[^{}]|\{[^{}]*\})*\}))*\}'

    def _make_bracket_pattern(self):
        """Creates pattern for optional parameters in square brackets"""
        return r'(?:\[(?:[^\[\]]|\[[^\[\]]*\])*\])?'

    def _initialize_separator_patterns(self):
        """Initialize the patterns for different granularity levels
        Latex is most common so we will use this as a base
        Custom ones are passed via the custom_separators parameter"""
        brace_pattern = self._make_brace_pattern()
        bracket_pattern = self._make_bracket_pattern()

        # Basic command pattern template
        cmd_pattern = lambda cmd: rf'\\{cmd}{bracket_pattern}\s*{brace_pattern}'

        self.separator_patterns = {
            Granularity.COARSE: {
                "chapter": cmd_pattern("chapter"),
                "section": cmd_pattern("section"),
                "subsection": cmd_pattern("subsection"),
                "subsubsection": cmd_pattern("subsubsection"),
            },

            Granularity.MEDIUM: {
                "paragraph": cmd_pattern("paragraph"),
                "subparagraph": cmd_pattern("subparagraph"),
                "part": cmd_pattern("part"),
            },

            Granularity.FINE: {
                "item": r'\\item\s+',
                "definition": cmd_pattern("definition"),
                "theorem": cmd_pattern("theorem"),
                "lemma": cmd_pattern("lemma"),
                "proof": cmd_pattern("proof"),
                "example": cmd_pattern("example"),
                "remark": cmd_pattern("remark"),
                "note": cmd_pattern("note"),
                "property": cmd_pattern("property"),
                "begin_enumerate": r'\\begin\{enumerate\}',
                "begin_itemize": r'\\begin\{itemize\}',
                "begin_description": r'\\begin\{description\}',
                "begin_figure": r'\\begin\{figure\}',
            },

            Granularity.FINEST: {
                "double_newline": r'\n\s*\n\s*',
                # "single_newline": r'\n'
            }
        }

        # Add any custom separators
        for granularity in Granularity:
            if granularity in self.custom_separators:
                self.separator_patterns[granularity].update(
                    self.custom_separators[granularity]
                )

    def _get_active_separators(self, granularity: Granularity) -> Dict[str, str]:
        """Get all separator patterns for given granularity and coarser levels"""
        active_separators = {}
        granularity_order = list(Granularity)
        start_idx = granularity_order.index(granularity)

        # Include patterns from current granularity and all coarser levels
        for idx in range(start_idx + 1):
            active_separators.update(
                self.separator_patterns[granularity_order[idx]]
            )

        return active_separators

    def _find_all_splits(self, text: str, separators: Dict[str, str]) -> List[Dict]:
        """Find all potential split points in the text using given separators."""
        splits = []

        for sep_type, pattern in separators.items():
            try:
                for match in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):
                    splits.append({
                        'position': match.start(),
                        'separator_type': sep_type,
                        'match': match.group(),
                        'length': len(match.group())
                    })
            except re.error as e:
                raise ValueError(f"Invalid regex pattern for {sep_type}: {e}")

        return sorted(splits, key=lambda x: x['position'])

    def _create_initial_chunks(self, text: str, splits: List[Dict], granularity: Granularity) -> List[Chunk]:
        """Create initial chunks based on split points."""
        chunks = []
        current_pos = 0
        for i, split in enumerate(splits):
            # End position is either the next split or the end of text
            end_pos = split['position']

            # Include the content up to the current separator in the chunk
            chunk_content = text[current_pos:end_pos].strip()
            if chunk_content:
                chunk = Chunk(
                    content=chunk_content,
                    start_pos=current_pos,
                    end_pos=end_pos,
                    separator_type=split['separator_type'],
                    granularity=granularity,
                    size=len(chunk_content)
                )
                chunks.append(chunk)

            # Update current_pos to the start of the current separator
            current_pos = end_pos

        # Handle any remaining text
        if current_pos < len(text):
            final_content = text[current_pos:].strip()
            if final_content:
                chunks.append(Chunk(
                    content=final_content,
                    start_pos=current_pos,
                    end_pos=len(text),
                    separator_type="final",
                    granularity=granularity,
                    size=len(final_content)
                ))
        return chunks

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks that are too small with subsequent chunks."""
        if not chunks:
            return chunks

        merged = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            combined_length = len(current_chunk.content) + len(next_chunk.content)

            if len(current_chunk.content) < self.min_chunk_length and combined_length <= self.max_chunk_length:
                current_chunk = Chunk(
                    content=current_chunk.content + "\n" + next_chunk.content,
                    start_pos=current_chunk.start_pos,
                    end_pos=next_chunk.end_pos,
                    separator_type=f"{current_chunk.separator_type}+{next_chunk.separator_type}",
                    granularity=current_chunk.granularity,
                    size=len(current_chunk.content + "\n" + next_chunk.content)
                )
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk

        merged.append(current_chunk)
        return merged

    def chunk_document(
            self,
            text: str,
            granularity: Granularity = Granularity.COARSE
    ) -> List[Chunk]:
        """
        Split LaTeX document into chunks based on specified granularity.


        :param text: Input LaTeX document text
        :param granularity: Granularity level for splitting
            COARSE: Major structural elements (chapter, section)
            MEDIUM: Minor structural elements (paragraph)
            FINE: List items, environments
            FINEST: Paragraphs (double newlines)
        :returns: List of Chunk objects containing the split document
        """
        # Get appropriate separators for this granularity level
        active_separators = self._get_active_separators(granularity)

        # Find all split points
        splits = self._find_all_splits(text, active_separators)

        if not splits:
            return [Chunk(
                content=text,
                start_pos=0,
                end_pos=len(text),
                separator_type="no_splits",
                granularity=granularity,
                size=len(text)
            )]

        # Create and merge chunks
        chunks = self._create_initial_chunks(text, splits, granularity)
        chunks = self._merge_small_chunks(chunks)

        return chunks


def clean_text(content):
    # TODO check if removing % comments is a good idea
    # also keep an eye out for more clutter
    begin_tag = r'\begin{document}'
    end_tag = r'\end{document}'

    begin_index = content.find(begin_tag)
    end_index = content.find(end_tag)

    if begin_index != -1 and end_index != -1:
        # Extract content between \begin{document} and \end{document}
        cleaned_content = content[begin_index + len(begin_tag):end_index].strip()
    else:
        # If tags are not found, return the entire content
        cleaned_content = content.strip()
    return cleaned_content


def merge_small_chunks(chunks: List[Chunk], min_size: int) -> List[Chunk]:
    """
    Merges consecutive chunks smaller than min_size into larger chunks.
    Processes the list from start to finish.


    :param chunks: List of Chunk objects to process
    :param min_size: Minimum size threshold for chunks

    :return: List of merged Chunk objects
    """
    if not chunks:
        return []

    result = []
    current_group = []
    current_size = 0

    for chunk in chunks:
        if chunk.size >= min_size:
            # If we have accumulated small chunks, merge them first
            if current_group:
                merged_chunk = Chunk(
                    content=''.join(c.content for c in current_group),
                    start_pos=current_group[0].start_pos,
                    end_pos=current_group[-1].end_pos,
                    separator_type=current_group[-1].separator_type,
                    granularity=current_group[0].granularity,
                    size=current_size
                )
                result.append(merged_chunk)
                current_group = []
                current_size = 0

            # Add the large chunk
            result.append(chunk)
        else:
            # Accumulate small chunks
            current_group.append(chunk)
            current_size += chunk.size

            # If accumulated size exceeds min_size, merge the group
            if current_size >= min_size:
                merged_chunk = Chunk(
                    content=''.join(c.content for c in current_group),
                    start_pos=current_group[0].start_pos,
                    end_pos=current_group[-1].end_pos,
                    separator_type=current_group[-1].separator_type,
                    granularity=current_group[0].granularity,
                    size=current_size
                )
                result.append(merged_chunk)
                current_group = []
                current_size = 0

    # Handle any remaining small chunks at the end
    if current_group:
        merged_chunk = Chunk(
            content=''.join(c.content for c in current_group),
            start_pos=current_group[0].start_pos,
            end_pos=current_group[-1].end_pos,
            separator_type=current_group[-1].separator_type,
            granularity=current_group[0].granularity,
            size=current_size
        )
        result.append(merged_chunk)

    return result


def automatic_chunking(latex_text: str, desired_chunk_size: int = 1000, hard_maximum_chunk_size: int = 2000,
                       hard_minimum_chunk_size=200,
                       chunking_per_granularity=None, fallback_strategy="split", try_clean_text=True,
                       correction_factor=1.2, document_type = "latex", custom_separator_patterns = None) -> List[Chunk]:
    """
    Automatically chunk a LaTeX document.

    Works recursively i.e. first splitting by sections etc. All chunks that are too large are then split further e.g. by paragraphs.

    Requires further processing as the chunks may still be too large

    fallback_strategy "split" means that the remaining strings will just split into smaller chunks until they are small enough.
    This will not take any structure into account!
    "ignore" will not do anything.
    "semantic" will use the sematic_chunking class.
    "split" (default) will simply cut using character count. If possible increase the hard limit instead of resorting to this.

    :param latex_text: The LaTeX document to chunk
    :param desired_chunk_size: The desired size of the chunks. Algorithm will go somewhere near this value for the mean. But the success heavily depends on the document!
    :param hard_maximum_chunk_size: The maximum size of a chunk. A chunk larger than this size will be referred to the fallback strategy
    :param hard_minimum_chunk_size: The minimum size of a chunk. A chunk smaller than this size will be merged with following chunks
    :param chunking_per_granularity: A list of tuples with the minimum and maximum chunk size for each granularity level
    :param fallback_strategy: The strategy to use when a chunk is too large. Options are "split", "ignore" and "semantic"
    :param try_clean_text: get rid of things outside of the document like imports
    :param correction_factor: A factor to correct the desired_chunk_size to get the mean chunk size close to the desired size
    :param document_type: The type of document. Choose from "latex" or "markdown".
    :param custom_separator_patterns: Custom separator patterns to use for chunking. Can be used to add support for documents other than LaTeX or Markdown.
    """
    desired_chunk_size *= correction_factor
    total_size = len(latex_text)

    if chunking_per_granularity is None:
        # when going into finer granularity we may split more than desired as we split by things that are no longer
        # logical entities. Hence the minimum chunk size is increases for each level to avoid this
        # Vice versa the maximum starts smaller to avoid splitting too little.
        chunking_per_granularity = [
            [int(desired_chunk_size * 0.2), max([int(desired_chunk_size * 1.5), hard_maximum_chunk_size])],
            [int(desired_chunk_size * 0.4), max([int(desired_chunk_size * 1.7), hard_maximum_chunk_size])],
            [int(desired_chunk_size * 0.5), max([int(desired_chunk_size * 1.8), hard_maximum_chunk_size])],
            [int(desired_chunk_size * 0.6), max([int(desired_chunk_size * 1.9), hard_maximum_chunk_size])]]
    granularity_list = [Granularity.COARSE, Granularity.MEDIUM, Granularity.FINE, Granularity.FINEST]
    kwargs = {}
    if custom_separator_patterns is not None and document_type!="markdown":
        kwargs["custom_separators"] = custom_separator_patterns
    elif document_type == "markdown":
        kwargs["custom_separators"] = markdown_separator_patterns
    chunker = RegexChunker(
        min_chunk_length=chunking_per_granularity[0][0],
        max_chunk_length=chunking_per_granularity[0][1],
        **kwargs
    )
    if try_clean_text:
        latex_text = clean_text(latex_text)
    chunks = chunker.chunk_document(latex_text, granularity_list[0])

    for granularity in granularity_list[1:]:
        for index in reversed(range(len(chunks))):
            chunker = RegexChunker(
                min_chunk_length=chunking_per_granularity[granularity_list.index(granularity)][0],
                max_chunk_length=chunking_per_granularity[granularity_list.index(granularity)][1]
            )
            chunk = chunks[index]
            if chunk.size < chunking_per_granularity[granularity_list.index(granularity)][0]:
                continue
            new_chunks = chunker.chunk_document(chunk.content, granularity)
            # need to insert new chunks in place of the old chunk
            chunks = chunks[:index] + new_chunks + chunks[index + 1:]
    # It can happen that there are too small elements as the latex chunker only considers previous elements for merging.
    # hence remants at e.g. the end of a section can appear. We will merge them back here
    chunks = merge_small_chunks(chunks, hard_minimum_chunk_size)

    too_large = False
    if fallback_strategy != "ignore":
        # check if any chunks are too large
        too_large = False
        for chunk in chunks:
            if chunk.size > hard_maximum_chunk_size:
                too_large = True
                break
    if too_large:
        print("Some chunks are too large. Applying fallback strategy. Consider checking the output")
        for index in reversed(range(len(chunks))):
            chunk = chunks[index]
            if chunk.size > hard_maximum_chunk_size:
                if fallback_strategy == "split":
                    divisor = int(chunk.size / desired_chunk_size)
                    cut = chunk.size // divisor
                    new_chunks = []
                    for i in range(divisor):
                        new_chunks.append(Chunk(content=chunk.content[i * cut:(i + 1) * cut],
                                                start_pos=chunk.start_pos + i * cut,
                                                end_pos=chunk.start_pos + (i + 1) * cut,
                                                separator_type="split",
                                                granularity=chunk.granularity,
                                                size=cut))
                    chunks = chunks[:index] + new_chunks + chunks[index + 1:]
    sizes = np.array([chunk.size for chunk in chunks])
    info_dict = {"total_size": total_size, "median_size": round(np.median(sizes)), "mean_size": round(np.mean(sizes)),
                 "max_size": np.max(sizes), "too_large": too_large}
    return chunks, info_dict


def print_chunks(chunks: List[Chunk]):
    """
    Print the whole document with the chunks separated by a line
    Use to check if the chunking is working as expected
    """
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk: {i}, Size: {chunk.size} ---")
        print(chunk.content)


def chunks_to_db_chunks_latex(chunks: List[Chunk], document_title, original_pdf=None, original_file=None) -> List[Dict[str, Union[str, int]]]:
    """
    Converts a list of Chunk objects from the latex parser into a format to be fed into the database.
    """

    db_chunks = []
    for chunk in chunks:
        metadata = {"size": chunk.size, "type": "text", "source_type": "latex",
                    "document_title": document_title}
        if original_file:
            metadata["original_file"] = original_file
        db_chunks.append({
            "text": chunk.content,
            "metadata": metadata,
            "id": generate_id(chunk.content)#hex(abs(hash(chunk.content)))[2:]
        })
    if original_pdf is not None:
        from . import pdf_page_matcher
        matcher = pdf_page_matcher.PDFLatexMatcher(original_pdf, chunks)
        page_assignments = matcher.match_chunks_to_pages()
        matcher.close()
        for i, chunk in enumerate(db_chunks):
            chunk["metadata"]["page"] = page_assignments[i]
    return db_chunks


def chunks_to_db_chunks_html(chunks: List[Chunk], title) -> List[Dict[str, Union[str, int]]]:
    """
    Converts a list of Chunk objects from the latex parser into a format to be fed into the database.
    """

    db_chunks = []
    for chunk in chunks:
        metadata = {"size":chunk.size, "type":"text", "source_type":"html",
                    "document_title": title}
        db_chunks.append({
            "text": chunk.content,
            "metadata": metadata,
            "id": generate_id(chunk.content)#hex(abs(hash(chunk.content)))[2:]
        })

    return db_chunks


def extract_heading(markdown_text: str) -> str:
    """
    Extract a heading from markdown text using various methods.
    Returns 'unknown' if no heading can be found.

    Methods tried (in order):
    1. Alternate heading syntax (=== or ---)
    2. Hash-style headers (#)
    3. First non-empty line
    """
    if not markdown_text or not isinstance(markdown_text, str):
        return "unknown"

    # Split into lines and remove empty ones
    lines = [line.strip() for line in markdown_text.splitlines()]
    lines = [line for line in lines if line]

    if not lines:
        return "unknown"

    # Method 1: Check for alternate heading syntax (=== or ---)
    for i in range(len(lines) - 1):
        current_line = lines[i]
        next_line = lines[i + 1]

        # Check if next line is all = or -
        if (set(next_line) == {'='} and len(next_line) >= 3) or \
                (set(next_line) == {'-'} and len(next_line) >= 3):
            return current_line

    # Method 2: Look for hash-style headers
    for line in lines:
        # Match any number of #s followed by space and text
        if line.startswith('#'):
            # Remove #s and spaces from start
            heading = line.lstrip('#').strip()
            if heading:  # Ensure there's text after the #s
                return heading

    # Method 3: Smart fallback - look for the first meaningful line
    for line in lines:
        # Skip likely non-heading lines
        if any(line.startswith(x) for x in ['>', '-', '*', '1.', '```', '    ']):
            continue
        # Skip lines that are too long (likely paragraphs)
        if len(line) > 100:
            continue
        # Skip lines that are all special characters
        if all(not c.isalnum() for c in line):
            continue
        return line

    return "unknown"


def clean_md(markdown):
    # get rid of excessive things that can appear either in the title or to weird conversion
    # especially long titles mess up character counting and we dont match for the full length anyway
    pattern = r'\n{4,}'
    markdown = re.sub(pattern, '\n\n\n', markdown)
    pattern = r'-{5,}'
    markdown = re.sub(pattern, '----', markdown)
    pattern = r'={5,}'
    markdown = re.sub(pattern, '====', markdown)
    return markdown