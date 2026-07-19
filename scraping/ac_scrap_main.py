import ac_movieid_scrapper as movies
import ac_reviews_scrapper as reviews

reviews.restore_db()

AC_LISTING_PAGES = [
    "https://www.allocine.fr/film/meilleurs/genre-13025/" # Example listing
]

# Main scraping loop
for listing in AC_LISTING_PAGES:
  print(f"Scraping for listing {listing}")
  for id in movies.scrap_from_listing(listing):
      reviews.scrap_and_dump(id)
