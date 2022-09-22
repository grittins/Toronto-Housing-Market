from bs4 import BeautifulSoup
import requests

baseurl = "https://trreb.ca/"
path = "index.php/market-news/market-watch/market-watch-archive"
_URL = baseurl + path

r = requests.get(_URL)

soup = BeautifulSoup(r.text)
urls = []
names = []
for i, link in enumerate(soup.findAll("a")):
    _FULLURL = (baseurl + str(link.get("href")))
    if _FULLURL.endswith(".pdf") and 'market-watch' in _FULLURL :
        urls.append(_FULLURL)
        names.append(soup.select("a")[i].attrs["href"])

names_urls = zip(names, urls)

for name, url in names_urls:
    print(url)
    r = requests.get(url)
    with open(r"C:\Users\Derek\Documents\Bootcamp\Module 20\sample data\pdfs" + name.split('/')[-1], "wb") as f:
        f.write(r.content)