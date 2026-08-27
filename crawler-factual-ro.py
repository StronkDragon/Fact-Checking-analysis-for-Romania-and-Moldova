import json
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

def fetch_all_from_endpoint(endpoint_url, dataset_name):
    print(f"\n--- Interogare endpoint: {endpoint_url} ---")
    dataset = []
    page = 1
    
    # 1. Facem primul request pentru a afla numărul total de pagini din header
    first_resp = requests.get(f"{endpoint_url}?per_page=100&page=1", headers=HEADERS, timeout=15)
    
    if first_resp.status_code != 200:
        print(f"Endpoint inaccesibil sau gol (Status: {first_resp.status_code}).")
        return []
    
    total_pages = int(first_resp.headers.get("X-WP-TotalPages", 1))
    total_entries = int(first_resp.headers.get("X-WP-Total", 0))
    print(f"Găsite {total_entries} înregistrări în total ({total_pages} pagini de descărcat).")
    
    # 2. Iterăm prin toate paginile
    for page in tqdm(range(1, total_pages + 1), desc=dataset_name):
        url = f"{endpoint_url}?per_page=100&page={page}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                break
                
            posts = resp.json()
            for post in posts:
                # Curățare titlu HTML
                raw_title = post.get("title", {}).get("rendered", "")
                title = BeautifulSoup(raw_title, "html.parser").get_text(strip=True)
                
                # Curățare text conținut
                raw_content = post.get("content", {}).get("rendered", "")
                content_text = BeautifulSoup(raw_content, "html.parser").get_text(separator=" ", strip=True)
                
                link = post.get("link", "")
                date = post.get("date", "")
                
                # Extragere preliminară verdict din primul cuvânt al titlului
                verdict = "unspecified"
                first_word = title.split()[0].lower().replace(":", "") if title else ""
                if first_word in ["fals", "adevărat", "adevarat", "trunchiat", "context", "parțial"]:
                    verdict = first_word

                dataset.append({
                    "source_country": "Romania",
                    "news_source": "Factual.ro",
                    "verdict": verdict,
                    "statement": title,
                    "statement_originator": "Unknown",
                    "statement_date": "",
                    "statement_source": "politics" if "/declaratii/" in link else "social_media",
                    "factchecker": "Factual.ro Team",
                    "factcheck_date": date,
                    "factcheck_analysis_link": link,
                    "analysis_text": content_text
                })
            
            time.sleep(0.3)  # Pauză scurtă între apeluri
        except Exception as e:
            print(f"Eroare la pagina {page}: {e}")
            break
            
    return dataset

def main():
    # Posibile endpoint-uri WordPress de pe Factual.ro
    endpoints = [
        #("https://www.factual.ro/wp-json/wp/v2/posts", "Factual Posts"),
        #("https://www.factual.ro/wp-json/wp/v2/declaratii", "Factual Declaratii"),
        ("https://www.factual.ro/wp-json/wp/v2/dezinformari-rs", "Factual Dezinformari Social Media"),
    ]
    
    all_results = []
    
    for url, label in endpoints:
        data = fetch_all_from_endpoint(url, label)
        all_results.extend(data)
        
    # Elimină eventualele duplicate după link
    unique_data = list({item["factcheck_analysis_link"]: item for item in all_results}.values())
    
    output_filename = "dataset_factual_api.json"
    output_filename_2 = "dataset_factual_api_2.json"
    
    with open(output_filename_2, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)
        
    print(f"\nFinalizat! Au fost descărcate {len(unique_data)} articole complete în '{output_filename}'.")

if __name__ == "__main__":
    main()
