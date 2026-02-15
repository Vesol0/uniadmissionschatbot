import cloudscraper
from bs4 import BeautifulSoup

class URLRetriever:
    def __init__(self):
        pass
    def getLinks(self, url: list):
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url)
        soup = BeautifulSoup(response.text, 'xml')
        links = [loc.text for loc in soup.find_all('loc') if '/courses/undergraduate-' in loc.text]
        return links




links = URLRetriever().getLinks("https://www.gcu.ac.uk/sitemap.xml")



for link in links:
    document = cloudscraper.create_scraper().get(link)
    soup = BeautifulSoup(document.content, 'html.parser')



    print(soup.find('title').text.split(' |')[0])

