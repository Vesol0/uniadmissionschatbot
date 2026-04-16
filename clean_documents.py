from bs4 import BeautifulSoup
import re
def document_cleaner(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    data = {}

    title_tag = soup.find('h1')
    data['Title'] = title_tag.text.strip() if title_tag else "Unknown Course"

    overview = soup.find(id="overview")
    data['Overview'] = overview.text if overview else ""
    study_sec = soup.find(id="what-will-you-study")
    if study_sec:
        data['Modules'] = ", ".join([li.text.strip() for li in study_sec.find_all('li')])

    info_sec = soup.find(id="course-information")
    data['Details'] = info_sec.text if info_sec else ""

    for key, value in data.items():
        if isinstance(value, str):
            data[key] = re.sub(r'\s+', ' ', value).strip()

    print(data)

    return data