import feedparser
import pandas as pd
import os

os.makedirs("raw", exist_ok=True)

feeds = {
    "moneycontrol": "https://www.moneycontrol.com/rss/marketsnews.xml",
    "economic_times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
}

for source, url in feeds.items():
    feed = feedparser.parse(url)
    print(f"\n{source}: {len(feed.entries)} entries found")

    if len(feed.entries) == 0:
        print(f"  ⚠️ Feed blocked or empty — saving empty CSV as placeholder")
        pd.DataFrame(columns=["date", "headline", "summary", "source"]).to_csv(
            f"raw/news_{source}_rss.csv", index=False
        )
        continue

    entries = []
    for entry in feed.entries:
        entries.append({
            "date": entry.get("published", ""),
            "headline": entry.get("title", ""),
            "summary": entry.get("summary", ""),
            "source": source
        })

    df = pd.DataFrame(entries)
    print(df[["date", "headline"]].head(3).to_string())
    df.to_csv(f"raw/news_{source}_rss.csv", index=False)
    print(f"Saved ✅")