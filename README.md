# AlloCiné Review Processing & NLP Pipeline

---

## Prerequisites & Setup

Before running the project, you need to install the dependencies and download the French NLP models.

### 1. Install Dependencies

Open your terminal in the project folder and run:

```bash
pip install -r requirements.txt

```

### 2. Download the French NLP Model

Since the preprocessing script uses **spaCy** to analyze French text, you must download the French language model. Run the following command:

```bash
python -m spacy download fr_core_news_sm

```

If spaCy complains that it cannot find pip or uv, you can bypass the helper and install the French model directly from the official GitHub repository using pip:
Bash

```bash
python -m pip install [https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.8.0/fr_core_news_sm-3.8.0-py3-none-any.whl](https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.8.0/fr_core_news_sm-3.8.0-py3-none-any.whl)
```

### 3. Gather data

To generate your dataset, you can either rely on the data already present in the `data` directory, or you can make it yourself using the scripts in the `scraping` directory.

The simplest and quickest way to pull a large amount of data is to choose a relevant movie listing from AlloCiné (e.g. `https://www.allocine.fr/film/meilleurs/`, or any other one linked from this page), and put it in the `AC_LISTING_REVIEWS` list in `ac_scraping_main.py`. You can use several listings at once, but beware that a single listing can take **several hours** to complete scraping. 

Then, simply run the script:

```bash
python scraping/ac_scrap_main.py
```

You can also scrap reviews from specific movies by using the `scrap_and_dump` function from `ac_reviews_scrapper.py`. 

#### Scripts breakdown
 - `ac_movieid_scrapper.py`: Exposes the `scrap_from_listing` function to extract all movie IDs from the provided listing page (usually, this yields 300 movies)
 - `ac_reviews_scrapper.py`: Exposes the `scrap_and_dump` function to extract all reviews from a single movie and save them in an SQLite DB and a JSON file. This is where all SQL functions live as well. All other functions are simply helpers for `scrap_and_dump` and should not be used outside of it. 
 - `ac_scrap_main.py`: Uses `scrap_from_listing` and `scrap_and_dump` to easily scrap a large amount of reviews.


---

## How to Run

Execute the main script from your terminal:

```bash
python main.py

```

### Adjusting Settings (Optional)

At the top of `main.py`, you can tweak these variables depending on your needs:

- `RUN_ON_SAMPLE = True`: Set to `True` to run a quick test on a subset of the data. Set to `False` to process **all** reviews in your JSON files.
- `SAMPLE_SIZE = 25000`: The number of reviews to process when running in sample mode.

---

## Output: Ready for Machine Learning!

Once the script finishes running, it will generate a file named:
**`data_processed.pkl`**

```python
import pandas as pd
df = pd.read_pickle("data_processed.pkl")

```
