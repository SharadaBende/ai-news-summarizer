"""Creates data/sample_references.csv: 5 articles with hand-written reference summaries (DEMO only)."""
import pandas as pd

df = pd.read_csv("data/sample_news.csv")
refs = {
    "clinched a dramatic victory": "The home team won a dramatic late match thanks to the striker's two goals. It now leads the league table by three points.",
    "told parliament that the new bill": "The prime minister said parliament would debate a new bill next week. Opposition leaders criticised the rush and a vote is expected before the session ends.",
    "latest smartphone with a faster processor": "The company unveiled a new smartphone with a faster chip and a better AI camera. It adds privacy features and goes on sale next month.",
    "raised interest rates by a quarter point": "The central bank raised interest rates by a quarter point to fight inflation. Economists expect slower borrowing and weaker growth, and stocks fell.",
    "reported a new treatment": "Doctors reported a new treatment that reduced symptoms in chronic disease patients during a trial. Regulators will now review the well tolerated drug.",
}
rows = []
for key, ref in refs.items():
    match = df[df["text"].str.contains(key, regex=False)]
    assert len(match) == 1, key
    rows.append({"text": match["text"].iloc[0], "reference": ref})
pd.DataFrame(rows).to_csv("data/sample_references.csv", index=False)
print("Wrote data/sample_references.csv")
