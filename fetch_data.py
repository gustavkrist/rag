# type: ignore
import json
import os

import pywikibot
from pywikibot import pagegenerators
from tqdm import tqdm

dirname = os.path.dirname(__file__)

pywikibot.config.family_files["terraria"] = os.path.join(
    dirname, "terraria_family"
)

site = pywikibot.Site("en", "terraria")
cat = pywikibot.Category(site, "Desktop content")
page_gen = pagegenerators.CategorizedPageGenerator(cat)
pages = list(page_gen)

articles = []
for page in tqdm(pages):
    if not (page.is_categorypage() or page.isDisambig() or page.isRedirectPage()):
        articles.append(
            {
                "id": str(page.pageid),
                "title": page.title(),
                "content": page.extract(variant="plain", intro=False),
            }
        )

os.makedirs(os.path.join(dirname, "data"), exist_ok=True)
with open(os.path.join(dirname, "data", "articles.json"), "w") as f:
    json.dump(articles, f)
