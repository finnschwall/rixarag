import re
import xml.etree.ElementTree as ET

def parse_wiki_xml(path):
    """
    Chunk a wikipedia xml file.
    These can be obtained from here https://en.wikipedia.org/wiki/Special:Export


    """
    tree = ET.parse(path)
    root = tree.getroot()
    entities = []
    for page in root[1:]:

        text = page.find("{http://www.mediawiki.org/xml/export-0.10/}revision").find(
            "{http://www.mediawiki.org/xml/export-0.10/}text").text
        title = page.find("{http://www.mediawiki.org/xml/export-0.10/}title").text
        id = page.find("{http://www.mediawiki.org/xml/export-0.10/}id").text
        if "Category:" in title:
            continue

        def repl(matchobj):
            hit = matchobj.groups()[0]
            full = matchobj.group()
            if "|" not in full or "efn|" in full:
                return ""
            elif "math| " in full:
                return f"${re.sub(r'{{((?:[^{}]|(?R))*)}}', repl, hit[6:])}$"
            elif "|" in hit:
                hit = re.sub(r"\|link=y", r"", full)
                if "10^|" in hit:
                    return f"10^{hit[6:-2]}"
                hit = re.sub(r"{{(.*?)\|(.*?)}}", r"\2", hit)
                return hit
            else:
                return full

        sections = re.split(r'={2,5}\s*(.*?)\s*={2,5}', text)
        headers = [title] + sections[1::2]
        section_text = sections[0::2]
        sections = {i: j for i, j in zip(headers, section_text)}
        entries_to_remove = (
            'See also', 'Footnotes', "References", "Sources", "History", "External links", "Bibliography")
        for k in entries_to_remove:
            sections.pop(k, None)

        for i in sections:
            text = sections[i]
            text = text.replace("&lt;", "<")
            text = text.replace("&gt;", ">")
            text = re.sub(r'\[\[(.*?)(?:\|.*?)?\]\]', r'\1', text)
            text = re.sub(r"<ref (.*?)>(.*?)</ref>", '', text)
            text = re.sub(r"<ref>(.*?)</ref>", '', text)
            text = re.sub(r"<ref (.*?)>", '', text)
            text = re.sub(r"<math(.*?)>(.*?)</math>", r'$\2$', text)
            text = re.sub(r"<sub>(.*?)</sub>", r'$\1$', text)
            text = re.sub(r"<sup>(.*?)</sup>", r'^{\1}', text)
            text = re.sub("&nbsp;", " ", text)
            text = re.sub("\t;", "", text)
            text = re.sub(r" {2,20}", "", text)
            text = re.sub(r'{{((?:[^{}]|(?R))*)}}', repl, text)
            text = re.sub("\n", "", text)  # <ref></ref>
            text = re.sub(r"<ref>(.*?)</ref>", '', text)
            text = re.sub(r"\'\'\'(.*?)\'\'\'", r"'\1'", text)
            text = re.sub(r"\'\'(.*?)\'\'", r"'\1'", text)
            entity = {"header": title, "content": i + ":\n" + text,
                      "url": f"https://en.wikipedia.org/?curid={id}#" + "_".join(i.split(" ")),
                      "subheader": i}
            entities.append(entity)
    return entities