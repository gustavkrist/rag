# type: ignore
import json
import os

import pywikibot
from pywikibot import pagegenerators
from tqdm import tqdm

dirname = os.path.dirname(__file__)

pywikibot.config.family_files["terraria"] = os.path.join(dirname, "terraria_family")

site = pywikibot.Site("en", "terraria")

categories = [
    "Boss NPCs",
    "Environments",
    "NPC NPCs",
    "Guides",
    "Liquids",
    "Debuffs",
    "Buffs",
    "Game mechanics",
    "Key items",
    "Boss summon items",
    "Permanent booster items",
]

pages = []
for cat_name in categories:
    cat = pywikibot.Category(site, cat_name)
    page_gen = pagegenerators.CategorizedPageGenerator(cat, recurse=False)
    pages.extend(page_gen)

pages = set(pages)

desktop_cat = pywikibot.Category(site, "Desktop content")
guide_cat = pywikibot.Category(site, "Guides")

articles = []
for page in tqdm(pages):
    if not (page.is_categorypage() or page.isDisambig() or page.isRedirectPage()):
        categories = list(page.categories())
        if desktop_cat in categories or guide_cat in categories:
            articles.append(
                {
                    "id": str(page.pageid),
                    "title": page.title(),
                    "content": page.extract(variant="plain", intro=False),
                    "url": page.full_url(),
                    "categories": [c.title() for c in categories],
                }
            )

os.makedirs(os.path.join(dirname, "data"), exist_ok=True)
with open(os.path.join(dirname, "data", "articles.json"), "w") as f:
    json.dump(articles, f)
