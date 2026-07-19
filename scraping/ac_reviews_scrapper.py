import json
import os
import re
import sqlite3
import time
import requests
from fake_useragent import UserAgent
from lxml import html

BASE_MOVIE_URL = "https://www.allocine.fr/film/fichefilm_gen_cfilm={movie_id}.html"
BASE_REVIEW_URL = "https://www.allocine.fr/film/fichefilm-{movie_id}/critiques/spectateurs?page={page_number}"
MAX_VALID_PAGE = 66666


# region SQLite functions
con = sqlite3.connect("reviews.db", isolation_level=None) # No ensure every query is committed directly, so we don't loose anything if the program crashes.
cur = con.cursor()

def restore_db():
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Movie(
      id TEXT PRIMARY KEY, 
      name TEXT, 
      release_date TEXT,
      runtime_in_minutes INTEGER,
      genres TEXT, -- comma separated
      synopsis TEXT
      );
  """)
    # No FOREIGN KEY to avoid useless errors, we don't *need* to force the relation
    # we can just fetch a failing movie manually if needed.
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Review(
      id TEXT PRIMARY KEY,
      movie_id TEXT,
      author TEXT,
      date TEXT,
      rating REAL,
      text_content TEXT,
      likes INT,
      dislikes INT
    )
  """)


def save_movie(movie):
    data = (
        str(movie["id"]),
        movie.get("name"),
        movie.get("release_date"),
        movie.get("runtime_in_minutes"),
        ",".join(movie.get("genres") or []),
        movie.get("synopsis"),
    )
    # INSERT OR REPLACE so re-saving a partially-filled-in movie (e.g. metadata
    # failed the first time but we still stored a stub row) doesn't blow up on
    # the PRIMARY KEY constraint.
    cur.execute("INSERT OR REPLACE INTO Movie VALUES (?,?,?,?,?,?)", data)


def save_review(review):
    data = (
        str(review["id"]),
        str(review["movie_id"]),
        review["author"],
        review["date"],
        review["rating"],
        review["text_content"],
        review["likes"],
        review["dislikes"],
    )
    cur.execute("INSERT OR REPLACE INTO Review VALUES (?,?,?,?,?,?,?,?)", data)


# endregion


# region Scraping functions

ua = UserAgent()
headers = {"User-Agent": ua.random}

def get_html(url):
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return html.fromstring(res.content)
    elif res.status_code == 429:
        print("Too many requests: Retrying in 30 seconds with new UA.")
        headers["User-Agent"] = ua.random
        time.sleep(30)
        return get_html(url)
    else:
      return html.fromstring("<html></html>")

def get_last_page_number(movie_id):
    url = BASE_REVIEW_URL.format(movie_id=movie_id, page_number=5000)
    res = requests.get(url)

    # When we request for a page number grater than the last page available on allociné,
    # the request gets redirected to the last page available. We can get its number by
    # following the redirects and see where when ended up
    end_location = res.url
    regex_search = re.search(r"\?page=([0-9]+)", end_location, re.IGNORECASE)
    if regex_search:
        return int(regex_search.group(1))
    return -1


MONTHS = {
    "janvier": "01",
    "février": "02",
    "mars": "03",
    "avril": "04",
    "mai": "05",
    "juin": "06",
    "juillet": "07",
    "août": "08",
    "septembre": "09",
    "octobre": "10",
    "novembre": "11",
    "décembre": "12",
}


def parse_date(string):
    if not string:
        return None
    date_search = re.search(r"([0-9]{1,2}) (\w+) ([0-9]{4})", string, re.IGNORECASE)

    if date_search:
        day = date_search.group(1)
        month = MONTHS.get(date_search.group(2).lower())
        year = date_search.group(3)
        if month:
            return f"{year}-{month}-{day}"
    return None


def parse_movie_metadata(string):
    parts = string.split("|")
    # Date is fetched before, we can ignore it
    runtime = parts[1] if len(parts) > 1 else ""
    genres = parts[2] if len(parts) > 2 else ""

    # == Runtime ==
    runtime_search = re.search(r"(\d+)h (\d+)min", runtime, re.IGNORECASE)
    if runtime_search:
        runtime = int(runtime_search.group(1)) * 60 + int(runtime_search.group(2))
    else:
        runtime = None

    # == Genres ==
    genres = [g.strip() for g in genres.split(",") if g.strip()]

    return {"runtime_in_minutes": runtime, "genres": genres}


def scrap_page(url, movie_id, verbose=True):
    dom = get_html(url)
    elements = dom.cssselect(".review-card")

    for el in elements:
        try:
            rating_el = el.cssselect(".stareval-note")
            rating = (
                float(rating_el[0].text.replace(",", "."))
                if rating_el and rating_el[0].text
                else None
            )

            content_el = el.cssselect(".content-txt.review-card-content")
            review_text = content_el[0].text if content_el else None

            # There can be more than one <a> tag there, but it's always the
            # first one that contains the username.
            review_id = el.get("id")
            username_el = el.cssselect(".meta .meta-title")
            username = username_el[0].text_content() if username_el else None

            date_el = el.cssselect(".review-card-meta-date")
            date = parse_date(date_el[0].text) if date_el else None

            useful_buttons = el.cssselect(
                ".review-card-social .reviews-users-comment-useful a"
            )
            likes = (
                int(useful_buttons[0].cssselect("span")[0].text)
                if len(useful_buttons) > 0
                else None
            )
            dislikes = (
                int(useful_buttons[1].cssselect("span")[0].text)
                if len(useful_buttons) > 1
                else None
            )

            review = {
                "id": review_id,
                "author": username,
                "date": date,
                "rating": rating,
                "text_content": review_text,
                "likes": likes,
                "dislikes": dislikes,
            }
        except Exception as e:
            if verbose:
                print(f"WARNING: Could not parse a review card on {url}: {e}")
            continue

        # Try to persist to sqlite, but a DB error (e.g. odd constraint issue)
        # shouldn't prevent us from still yielding/returning the data in the
        # final JSON.
        try:
            save_review({**review, "movie_id": movie_id})
        except Exception as e:
            if verbose:
                print(f"WARNING: Could not save review {review.get('id')} to DB: {e}")

        yield review


def scrap_reviews(movie_id, verbose=True):
    last_page = get_last_page_number(movie_id)

    if verbose:
        print(f"Parsing up to {last_page} pages for movie {movie_id}...")

    for i in range(1, last_page + 1):
        time.sleep(1)
        page_url = BASE_REVIEW_URL.format(movie_id=movie_id, page_number=i)
        if verbose:
            print(f"Parsing page {i}/{last_page}...")
        try:
            yield from scrap_page(page_url, movie_id, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"WARNING: Could not parse page {i} ({page_url}): {e}")
            continue


def scrap_movie(movie_id, verbose=True):
    dataset = {
        "id": movie_id,
        "name": None,
        "release_date": None,
        "runtime_in_minutes": None,
        "genres": [],
        "synopsis": None,
        "reviews": [],
    }

    # == MOVIE INFO ==
    try:
        if verbose:
            print(f"Parsing metadata for movie {movie_id}")
        url = BASE_MOVIE_URL.format(movie_id=movie_id)
        dom = get_html(url)

        movie_name = dom.cssselect(".titlebar")[0].text_content()
        movie_release_date = parse_date(
            dom.cssselect(".meta-body .date")[0].text_content()
        )
        movie_meta = parse_movie_metadata(
            dom.cssselect(".meta-body-info")[0].text_content()
        )
        movie_synopsis = dom.cssselect("#synopsis-details .content-txt")[
            0
        ].text_content()

        dataset.update(
            {
                "name": movie_name,
                "release_date": movie_release_date,
                "runtime_in_minutes": movie_meta["runtime_in_minutes"],
                "genres": movie_meta["genres"],
                "synopsis": movie_synopsis,
            }
        )
    except Exception as e:
        print(f"WARNING: Could not parse metadata for movie {movie_id}: {e}")

    try:
        save_movie(dataset)
    except Exception as e:
        print(f"WARNING: Could not save movie {movie_id} to DB: {e}")

    # == REVIEWS ==
    try:
        for review in scrap_reviews(movie_id, verbose=verbose):
            dataset["reviews"].append(review)
    except Exception as e:
        # scrap_reviews already catches per-page/per-review errors internally;
        # this is a last-resort net in case something unexpected still
        # escapes it (e.g. a KeyboardInterrupt-adjacent condition).
        print(f"WARNING: Reviews scraping stopped early for movie {movie_id}: {e}")

    if verbose:
        print(
            f"Finished parsing {len(dataset['reviews'])} reviews for movie {movie_id}."
        )

    return dataset


def scrap_and_dump(movie_id, dump_output="data"):
    """Scrape a movie and always write out whatever data was collected,
    even if scraping failed partway through or raised unexpectedly."""
    dataset = {"id": movie_id, "reviews": []}
    try:
        dataset = scrap_movie(movie_id)
    except Exception as e:
        # scrap_movie is written to not raise, but this is a final safety
        # net so a JSON dump is *guaranteed* no matter what goes wrong.
        print(f"ERROR: Unexpected failure while scraping movie {movie_id}: {e}")
    finally:
        os.makedirs("dump", exist_ok=True)
        out_path = f"{dump_output}/{movie_id}.json"
        try:
            with open(out_path, "w") as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)
            print(
                f"Saved data to {out_path} "
                f"({len(dataset.get('reviews', []))} reviews collected)."
            )
        except Exception as e:
            print(f"ERROR: Could not write JSON dump for movie {movie_id}: {e}")

    return dataset


# endregion