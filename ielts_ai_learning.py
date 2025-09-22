import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import json
import re

# Paths
DATA_DIR = "./ielts_data"
DATA_PATH = os.path.join(DATA_DIR, "learning_items.csv")
RECOMMENDER_PATH = os.path.join(DATA_DIR, "recommender.joblib")
BAND_MODEL_PATH = os.path.join(DATA_DIR, "band_model.joblib")
SYNTH_BAND_MODEL_PATH = os.path.join(DATA_DIR, "synth_band_model.joblib")

os.makedirs(DATA_DIR, exist_ok=True)
recommender, vectorizer = joblib.load("recommender.joblib")
band_model = joblib.load("band_model.joblib")

# Utilities: simple text features for essays/speaking transcripts
def extract_text_features(text):
    # normalize
    text = text.strip()
    words = re.findall(r'\b\w+\b', text)
    if len(words) == 0:
        return {
            'word_count': 0,
            'sent_count': 0,
            'avg_sent_len': 0.0,
            'uniq_ratio': 0.0,
            'avg_word_len': 0.0,
            'punct_ratio': 0.0
        }
    sents = re.split(r'[.!?]+', text)
    sents = [s.strip() for s in sents if s.strip()]
    word_count = len(words)
    sent_count = len(sents)
    avg_sent_len = word_count / sent_count if sent_count>0 else word_count
    uniq_ratio = len(set([w.lower() for w in words])) / word_count
    avg_word_len = np.mean([len(w) for w in words])
    punct_ratio = sum(1 for c in text if c in '.,;:!?') / max(1, len(text))
    return {
        'word_count': word_count,
        'sent_count': sent_count,
        'avg_sent_len': avg_sent_len,
        'uniq_ratio': uniq_ratio,
        'avg_word_len': avg_word_len,
        'punct_ratio': punct_ratio
    }

# Load dataset (packaged) or create a tiny fallback
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    # fallback sample if dataset missing
    rows = [
        {"item_id":"ITEM001","title":"Reading Easy","topic":"IELTS Reading: Skimming & Scanning","difficulty":"easy","content":"Simple reading practice.","sample_question":"Find the main idea.","sample_answer":"Main idea is ..."},
        {"item_id":"ITEM002","title":"Writing Medium","topic":"IELTS Writing Task 2: Opinion Essays","difficulty":"medium","content":"Opinion essay practice.","sample_question":"Write 250 words.","sample_answer":"Model essay excerpt ..."}
    ]
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    return df

# Train/load recommender (TF-IDF + NearestNeighbors)
def train_recommender(df):
    vec = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vec.fit_transform(df['content'].fillna(''))
    nn = NearestNeighbors(n_neighbors=6, metric='cosine')
    nn.fit(X)
    joblib.dump({'vec':vec,'nn':nn,'items':df}, RECOMMENDER_PATH)
    return {'vec':vec,'nn':nn,'items':df}

def load_or_train_recommender():
    if os.path.exists(RECOMMENDER_PATH):
        try:
            return joblib.load(RECOMMENDER_PATH)
        except Exception:
            pass
    df = load_dataset()
    return train_recommender(df)

# Create a synthetic dataset of essay features -> band for demonstration
def create_synthetic_band_dataset(n=500):
    rng = np.random.RandomState(42)
    rows = []
    for _ in range(n):
        # generate plausible feature ranges
        word_count = rng.randint(100, 450)  # Task 2 essays typically 250+, but we simulate variety
        sent_count = rng.randint(5, 18)
        avg_sent_len = word_count / sent_count
        uniq_ratio = rng.uniform(0.35, 0.8)
        avg_word_len = rng.uniform(3.5, 6.0)
        punct_ratio = rng.uniform(0.01, 0.07)
        # synthetic band: base + contributions from features
        band = 4.0 + (word_count - 100)/350 * 1.5 + (uniq_ratio - 0.35)/0.45 * 2.0 + (avg_sent_len - 8)/10 * 1.0
        band += rng.normal(0, 0.4)
        band = float(max(3.0, min(9.0, band)))
        rows.append({
            'word_count':word_count,
            'sent_count':sent_count,
            'avg_sent_len':avg_sent_len,
            'uniq_ratio':uniq_ratio,
            'avg_word_len':avg_word_len,
            'punct_ratio':punct_ratio,
            'band':band
        })
    return pd.DataFrame(rows)

def train_synthetic_band_model(df_features=None):
    if df_features is None:
        df_features = create_synthetic_band_dataset(600)
    X = df_features[['word_count','sent_count','avg_sent_len','uniq_ratio','avg_word_len','punct_ratio']].values
    y = df_features['band'].values
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X, y)
    joblib.dump({'model':model}, SYNTH_BAND_MODEL_PATH)
    return model

def load_or_train_band_model():
    if os.path.exists(SYNTH_BAND_MODEL_PATH):
        try:
            return joblib.load(SYNTH_BAND_MODEL_PATH)['model']
        except Exception:
            pass
    return train_synthetic_band_model()

# Recommendation helper for a profile (simple)
def recommend_for_profile(profile, model_pack, top_k=6, prefer_difficulty=None):
    items = model_pack['items']
    vec = model_pack['vec']
    profs = profile['proficiency'] if 'proficiency' in profile else {}
    topic_order = sorted(profs.items(), key=lambda kv: kv[1]) if profs else []
    weak_topics = [t for t,_ in topic_order[:3]] if topic_order else []
    # fallback: recommend by topic if available
    if weak_topics:
        subset = items[items['topic'].isin(weak_topics)]
        return subset.head(top_k)
    return items.head(top_k)

# ----------------------
# App UI
# ----------------------
st.set_page_config(page_title="IELTS AI Personalized Learning", layout="wide")
st.title("IELTS AI — Personalized Learning with Band Feedback (Synthetic)")

# Load resources
items_df = load_dataset()
model_pack = load_or_train_recommender()
band_model = load_or_train_band_model()

# Session state for profiles
if 'profiles' not in st.session_state:
    st.session_state['profiles'] = {}

# Sidebar login
with st.sidebar:
    st.header("User / Instructor")
    mode = st.selectbox("Mode", ["Student","Instructor"])
    email = st.text_input("Email (student profile)")
    name = st.text_input("Name", value="Student")
    if st.button("Load profile"):
        if not email:
            st.warning("Enter email to load/create profile.")
        else:
            if email not in st.session_state['profiles']:
                # initialize profile with proficiency per topic (0-1)
                topics = sorted(items_df['topic'].unique().tolist())
                prof = {t:0.3 for t in topics}
                st.session_state['profiles'][email] = {'display_name':name,'proficiency':prof,'history':[],'bands':[]}
            st.success(f"Profile for {st.session_state['profiles'][email]['display_name']} loaded.")

# Main layout
if mode == "Student":
    st.subheader("Student Dashboard")
    if not email or email not in st.session_state['profiles']:
        st.info("Load your profile from the sidebar (enter your email and click 'Load profile').")
    else:
        profile = st.session_state['profiles'][email]
        st.markdown(f"**Welcome — {profile['display_name']}**")
        # Recommendations
        st.markdown("### Personalized recommendations")
        pref_diff = st.selectbox("Preferred difficulty", ["any","easy","medium","hard"], index=0)
        pref = None if pref_diff=="any" else pref_diff
        recs = recommend_for_profile(profile, model_pack, top_k=6, prefer_difficulty=pref)
        for _, r in recs.iterrows():
            with st.expander(f"{r['title']} — {r['difficulty']}"):
                st.write(r['content'])
                if 'sample_question' in r and not pd.isna(r['sample_question']):
                    st.write("**Practice:**", r["sample_question"])
                    st.write("**Model answer:**", r.get('sample_answer',''))

        st.markdown('---')
        st.markdown('### Band-score feedback (Essay / Speaking transcript)')
        text = st.text_area("Paste your Writing Task 2 essay or speaking transcript here (250+ words recommended)")
        if st.button("Analyze text"):
            if not text.strip():
                st.warning("Please paste some text to analyze.")
            else:
                feats = extract_text_features(text)
                X = np.array([[feats['word_count'], feats['sent_count'], feats['avg_sent_len'],
                               feats['uniq_ratio'], feats['avg_word_len'], feats['punct_ratio']]])
                try:
                    pred = band_model.predict(X)[0]
                    pred = float(max(3.0, min(9.0, pred)))
                except Exception:
                    # fallback: simple heuristic mapping
                    pred = 4.0 + (feats['uniq_ratio']-0.35)/0.65*3.0
                    pred = float(max(3.0, min(9.0, pred)))
                st.success(f"Predicted band (synthetic): {pred:.1f}")
                # Strengths & weaknesses (simple rule-based messages)
                msgs = []
                if feats['word_count'] < 200:
                    msgs.append("Essay is short — aim for 250+ words for Task 2.")
                else:
                    msgs.append("Good essay length.")
                if feats['uniq_ratio'] < 0.45:
                    msgs.append("Limited vocabulary range — try to use more varied words.")
                else:
                    msgs.append("Good lexical variety.")
                if feats['avg_sent_len'] < 10:
                    msgs.append("Short sentences — add complex/compound sentences for a higher band.")
                else:
                    msgs.append("Good sentence complexity.")
                st.markdown('**Feedback**')
                for m in msgs:
                    st.write('-', m)
                tips = ["Use linking words (however, moreover, therefore)", "Add one complex sentence per paragraph", "Paraphrase the question to show lexical resource"]
                st.markdown('**Suggested tip:**')
                st.write('-', np.random.choice(tips))
                # store in profile
                profile['history'].append({'text_sample': text[:200], 'predicted_band': float(pred)})
                profile['bands'].append(float(pred))
                st.session_state['profiles'][email] = profile

        st.markdown('---')
        st.markdown('### Your profile & progress')
        st.write('Estimated bands from submitted texts:')
        if profile.get('bands'):
            st.line_chart(pd.Series(profile['bands']))
        else:
            st.info('No band predictions yet. Submit an essay to get started.')
        st.write('Proficiency per topic:')
        prof_df = pd.DataFrame(list(profile['proficiency'].items()), columns=['topic','score']).sort_values('score')
        st.table(prof_df)

else:
    # Instructor view
    st.subheader("Instructor Dashboard")
    st.markdown('### Dataset & Recommender')
    st.write(f'Items: {len(items_df)}')
    if st.button("Retrain recommender"):
        train_recommender(items_df)
        st.success("Recommender retrained.")
    if st.button("Retrain synthetic band model"):
        train_synthetic_band_model()
        st.success("Synthetic band model retrained.")

    st.markdown('---')
    st.markdown('### Add new practice item')
    with st.form('add_item'):
        title = st.text_input('Title')
        topic = st.text_input('Topic (use IELTS topic labels)')
        diff = st.selectbox('Difficulty', ['easy','medium','hard'])
        content = st.text_area('Content', height=150)
        sample_q = st.text_input('Sample question')
        sample_a = st.text_input('Sample answer')
        submitted = st.form_submit_button('Add item')
        if submitted:
            df = pd.read_csv(DATA_PATH)
            new_id = f"ITEM{len(df)+1:03d}"
            new_row = {'item_id':new_id,'title':title,'topic':topic,'difficulty':diff,'content':content,
                       'sample_question':sample_q,'sample_answer':sample_a}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(DATA_PATH, index=False)
            st.success('Item added. Retrain recommender to include it.')

    st.markdown('---')
    st.markdown('### Student profiles (export)')
    profiles = st.session_state.get('profiles', {})
    st.write(f'Stored profiles in session: {len(profiles)}')
    if profiles:
        # show a summary table
        rows = []
        for k,v in profiles.items():
            rows.append({'email':k,'name':v.get('display_name',''),'bands':len(v.get('bands',[]))})
        st.table(pd.DataFrame(rows))
        if st.button('Export session profiles as JSON'):
            st.download_button('Download profiles', data=json.dumps(profiles, indent=2), file_name='session_profiles.json')