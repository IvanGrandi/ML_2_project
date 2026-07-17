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

---

## How to Run

Execute the main script from your terminal:

```bash
python main.py

```

### Adjusting Settings (Optional)

At the top of `main.py`, you can tweak these variables depending on your needs:

- `EXECUTER_SUR_ECHANTILLON = True`: Set to `True` to run a quick test on a subset of the data. Set to `False` to process **all** reviews in your JSON files.
- `TAILLE_ECHANTILLON = 25000`: The number of reviews to process when running in sample mode.

---

## Output: Ready for Machine Learning!

Once the script finishes running, it will generate a file named:
**`data_processed.pkl`**

```python
import pandas as pd
df = pd.read_pickle("data_processed.pkl")

```
