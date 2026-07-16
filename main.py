import json
import pandas as pd
import preprocessing as prep

#with open("allocine_reviews.json", "r", encoding="utf-8") as f:
#    data = json.load(f)

#df = pd.json_normalize(data)

#print(df.columns)

if __name__ == "__main__":
    exemple_review = "Les acteurs étaient parfaits... &amp; j'ai adoré la fin ! http://allocine.fr"
    res = prep.get_cleaning_steps(exemple_review)

    for step, result in res.items():
        print(f"{step} :\n   {result}\n")