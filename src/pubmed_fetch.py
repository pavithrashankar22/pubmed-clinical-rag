import json
import time
import os
from Bio import Entrez

Entrez.email = "pavithra22pavz@gmail.com"

SEARCH_QUERY = "AI hallucination medical safety clinical"
MAX_ARTICLES = 80


def fetch_pubmed_ids(query, max_results):
    print(f"Searching PubMed for: '{query}'...")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    ids = record["IdList"]
    print(f"Found {len(ids)} articles.")
    return ids


def fetch_abstracts(pmids):
    print("Fetching abstracts...")
    articles = []
    batch_size = 20

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(batch),
            rettype="xml",
            retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()

        for record in records["PubmedArticle"]:
            try:
                article = record["MedlineCitation"]["Article"]

                title = str(article.get("ArticleTitle", "No title"))

                abstract_data = article.get("Abstract", {})
                abstract_texts = abstract_data.get("AbstractText", [])
                if isinstance(abstract_texts, list):
                    abstract = " ".join(str(t) for t in abstract_texts)
                else:
                    abstract = str(abstract_texts)

                if not abstract or len(abstract) < 50:
                    continue

                pub_date = article.get("Journal", {}).get(
                    "JournalIssue", {}
                ).get("PubDate", {})
                year = pub_date.get("Year", "Unknown")

                pmid = str(record["MedlineCitation"]["PMID"])

                # Clean text — remove non-ascii characters
                title    = title.encode("ascii", "ignore").decode()
                abstract = abstract.encode("ascii", "ignore").decode()

                articles.append({
                    "pmid":     pmid,
                    "title":    title,
                    "abstract": abstract,
                    "year":     year,
                    "source":   f"PubMed PMID {pmid} ({year}): {title}"
                })

            except Exception as e:
                print(f"  Skipping malformed record: {e}")
                continue

        time.sleep(0.4)
        print(f"  Fetched {min(i + batch_size, len(pmids))}/{len(pmids)}...")

    return articles


def save_articles(articles, output_path):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(articles)} articles to {output_path}")


if __name__ == "__main__":
    # Load existing articles
    existing = []
    if os.path.exists("data/abstracts.json"):
        with open("data/abstracts.json", "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing articles")

    # Fetch new ones
    pmids    = fetch_pubmed_ids(SEARCH_QUERY, MAX_ARTICLES)
    articles = fetch_abstracts(pmids)

    # Combine — deduplicate by PMID
    all_pmids = {a["pmid"] for a in existing}
    new_only  = [a for a in articles if a["pmid"] not in all_pmids]
    combined  = existing + new_only

    save_articles(combined, "data/abstracts.json")
    print(f"Total articles now: {len(combined)}")
    print(f"New articles added: {len(new_only)}")