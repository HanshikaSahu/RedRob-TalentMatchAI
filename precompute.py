import json, numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, date

model = SentenceTransformer("all-MiniLM-L6-v2")
consulting = ["TCS","Infosys","Wipro","Accenture","Cognizant","Capgemini"]

def is_disqualified(c):
    companies = [r["company"] for r in c["career_history"]]
    if all(any(con in comp for con in consulting) for comp in companies):
        return True
    last_active = datetime.strptime(
        c["redrob_signals"]["last_active_date"], "%Y-%m-%d").date()
    if (date.today() - last_active).days > 180:
        return True
    return False

ids = []
texts = []
survivor_profiles = []

print("Reading candidates...")
with open("candidates.jsonl", "r") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        c = json.loads(line)
        if is_disqualified(c):
            del c
            continue
        career_text = " ".join([r["description"] for r in c["career_history"]])
        skills_text = " ".join([s["name"] for s in c["skills"]])
        ids.append(c["candidate_id"])
        texts.append(career_text + " " + skills_text)
        survivor_profiles.append(c)
        if i % 10000 == 0:
            print(f"Read {i}/100000...")

print(f"Encoding {len(texts)} survivors...")
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

np.save("embeddings.npy", embeddings)

with open("survivor_ids.json", "w") as f:
    json.dump(ids, f)

with open("survivor_profiles.json", "w") as f:
    json.dump(survivor_profiles, f)

print(f"Done! Saved {len(ids)} candidates to disk.")
print("Files created: embeddings.npy, survivor_ids.json, survivor_profiles.json")