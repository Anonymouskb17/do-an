import requests
import random

def search_items(keyword, limit=50):
    url = "https://shopee.vn/api/v4/search/search_items"
    params = {
        "by": "relevancy",
        "keyword": keyword,
        "limit": limit,
        "newest": 0,
        "order": "desc"
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    items = r.json()["items"]

    results = []
    for it in items:
        basic = it["item_basic"]
        results.append({
            "shopid": basic["shopid"],
            "itemid": basic["itemid"],
            "name": basic["name"]
        })

    return results

# Ví dụ dùng
items = search_items("áo thun", limit=50)
random_item = random.choice(items)

print(random_item)

