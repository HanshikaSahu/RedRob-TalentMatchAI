import argparse, json, numpy as np, csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, date

parser = argparse.ArgumentParser()
parser.add_argument("--candidates", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

JD_TEXT = """
Senior AI Engineer with production experience in embeddings-based retrieval systems,
vector databases, hybrid search, ranking evaluation metrics like NDCG and MRR,
strong Python, applied ML at product companies, LLM fine-tuning with LoRA or QLoRA,
building recommendation and ranking systems, shipping end-to-end ML systems to real users,
evaluation frameworks for ranking, A/B testing, retrieval quality measurement.
"""
jd_embedding = model.encode([JD_TEXT])

print("Loading pre-computed embeddings...")
embeddings = np.load("embeddings.npy")
with open("survivor_ids.json") as f:
    survivor_ids = json.load(f)

print("Loading survivor profiles...")
with open("survivor_profiles.json") as f:
    survivor_profiles = json.load(f)
candidate_lookup = {c["candidate_id"]: c for c in survivor_profiles}

print(f"Loaded {len(survivor_ids)} candidates")

print("Computing similarities...")
scores_raw = cosine_similarity(jd_embedding, embeddings)[0]

def score_candidate(c, semantic_score):
    sig = c["redrob_signals"]
    career_text = " ".join([r["description"] for r in c["career_history"]])

    if semantic_score < 0.28:
        return 0.0

    ML_KEYWORDS = ["retrieval","ranking","recommendation","embeddings",
                   "vector","nlp","fine-tun","transformer","search"]
    ml_hits = sum(1 for kw in ML_KEYWORDS if kw in career_text.lower())
    semantic_score = min(semantic_score + ml_hits * 0.03, 1.0)

    IRRELEVANT_TITLES = ["graphic","civil","mechanical","accountant","marketing",
                         "operations","mobile","project manager","customer support",
                         "business analyst","qa","frontend","java",".net"]
    title = c["profile"]["current_title"].lower()
    title_penalty = 0.25 if any(t in title for t in IRRELEVANT_TITLES) else 1.0

    consulting = ["TCS","Infosys","Wipro","Accenture","Cognizant","Capgemini"]
    exp_score = 0
    yoe = c["profile"]["years_of_experience"]
    if 5 <= yoe <= 9: exp_score += 0.4
    elif 4 <= yoe <= 11: exp_score += 0.2
    for role in c["career_history"]:
        if not any(con in role["company"] for con in consulting):
            exp_score += 0.1
    exp_score = min(exp_score, 1.0)

    avail_score = 0
    if sig["open_to_work_flag"]: avail_score += 0.3
    last_active = datetime.strptime(sig["last_active_date"], "%Y-%m-%d").date()
    days_inactive = (date.today() - last_active).days
    if days_inactive < 30: avail_score += 0.3
    elif days_inactive < 90: avail_score += 0.2
    avail_score += sig["recruiter_response_rate"] * 0.2
    if sig["notice_period_days"] <= 30: avail_score += 0.2
    elif sig["notice_period_days"] <= 60: avail_score += 0.1
    avail_score = min(avail_score, 1.0)

    verify_score = 0
    assessed = sig["skill_assessment_scores"]
    if assessed:
        verify_score += (sum(assessed.values()) / len(assessed) / 100) * 0.5
    if sig["github_activity_score"] > 0:
        verify_score += (sig["github_activity_score"] / 100) * 0.3
    if sig["verified_email"] and sig["verified_phone"]: verify_score += 0.2

    bonus_score = 0
    nice_skills = ["lora","qlora","peft","fine-tuning","open-source",
                   "learning to rank","xgboost","hr-tech","recruiting"]
    candidate_skills_lower = [s["name"].lower() for s in c["skills"]]
    for skill in nice_skills:
        if any(skill in s for s in candidate_skills_lower): bonus_score += 0.15
    bonus_score = min(bonus_score, 1.0)

    short_stints = [r for r in c["career_history"]
                    if r["duration_months"] < 12 and not r["is_current"]]
    hopper_penalty = 1.0
    if len(short_stints) >= 2: hopper_penalty = 0.7
    if len(short_stints) >= 3: hopper_penalty = 0.5

    final = (semantic_score*0.35 + exp_score*0.25 +
             avail_score*0.20 + verify_score*0.12 + bonus_score*0.08)
    return round(final * title_penalty * hopper_penalty, 4)

def generate_reasoning(c, rank):
    sig = c["redrob_signals"]
    profile = c["profile"]
    title = profile["current_title"]
    yoe = profile["years_of_experience"]
    company = profile["current_company"]
    assessed = sig["skill_assessment_scores"]
    career_text = " ".join([r["description"] for r in c["career_history"]]).lower()
    reasoning = f"{title} with {yoe} years experience at {company}"
    if assessed:
        best_skill = max(assessed, key=assessed.get)
        reasoning += f"; verified {best_skill} score {assessed[best_skill]:.0f}/100"
    ml_signals = [kw for kw in ["retrieval","ranking","recommendation","embeddings","nlp"]
                  if kw in career_text]
    if ml_signals:
        reasoning += f"; hands-on {', '.join(ml_signals[:2])} work in career history"
    concerns = []
    if sig["notice_period_days"] > 60: concerns.append(f"notice period {sig['notice_period_days']} days")
    if sig["recruiter_response_rate"] < 0.3: concerns.append("low response rate")
    if rank > 50: concerns.append("borderline fit")
    if concerns: reasoning += f". Concern: {'; '.join(concerns)}."
    else: reasoning += "."
    return reasoning

print("Scoring candidates...")
results = []
survivor_id_set = set(survivor_ids)
for i, cid in enumerate(survivor_ids):
    c = candidate_lookup.get(cid)
    if not c:
        continue
    score = score_candidate(c, float(scores_raw[i]))
    if score > 0:
        results.append((score, cid, c))

results.sort(key=lambda x: (-x[0], x[1]))
top100 = results[:100]

with open(args.out, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for i, (score, cid, c) in enumerate(top100, 1):
        writer.writerow([cid, i, score, generate_reasoning(c, i)])

print(f"Done! Saved {args.out} with {len(top100)} candidates")