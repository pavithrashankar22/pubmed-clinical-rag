import json
import time
import os
from Bio import Entrez

Entrez.email = "pavithra22pavz@gmail.com"

SEARCH_QUERY = "large language models clinical decision support"
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

                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "source": f"PubMed PMID {pmid} ({year}): {title}"
                })

            except Exception as e:
                print(f"  Skipping malformed record: {e}")
                continue

        time.sleep(0.4)
        print(f"  Fetched {min(i + batch_size, len(pmids))}/{len(pmids)}...")

    return articles

def save_articles(articles, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(articles)} articles to {output_path}")


if __name__ == "__main__":
    pmids    = fetch_pubmed_ids(SEARCH_QUERY, MAX_ARTICLES)
    articles = fetch_abstracts(pmids)
    save_articles(articles, "data/abstracts.json")

    if articles:
        print("\n── Sample Article ──")
        print(f"Title:    {articles[0]['title']}")
        print(f"PMID:     {articles[0]['pmid']}")
        print(f"Abstract: {articles[0]['abstract'][:200]}...")