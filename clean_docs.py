from bs4 import BeautifulSoup
import regex as re


def clean_doc(doc):
    soup = BeautifulSoup(doc, 'html.parser')
    for tag in soup(['script', 'style', 'footer']):
        tag.decompose()

    return soup
def get_relevant_content(soup: BeautifulSoup):
    content = soup.find('main')
    if content is None:
        content = soup.body

    text = content.get_text(separator="\n", strip=True)
    return text

def clean_content(text):
    text = re.sub("r\s+", " ", text)
    text = re.sub("\n+", "\n", text)
    return text.strip()
