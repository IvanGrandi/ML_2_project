from ac_reviews_scrapper import get_html
import re

def parse_id_from_link(link):
  search = re.search(r"fichefilm-(\d+)", link)
  if search:
    return int(search.group(1))
  return None
  

def scrap_page(url):
  dom = get_html(url)
  elements = dom.cssselect('li.mdl')

  for el in elements:
    a = el.cssselect('a.rating-title')[0]
    link = a.get("href")
    if link:
      yield parse_id_from_link(link)


def scrap_from_listing(listing_url):
  for i in range(1, 30): # There are always 30 pages per movie listing on AlloCiné (expect for too recent dated listing)
    url = listing_url + f'?page={i}'
    print(f"Page {i}/30 of listing {listing_url}")
    for id in scrap_page(url):
      print(f"Found: {id}")
      yield id