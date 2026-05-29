<h1 align="center">SkySense ✈️</h1>
<h3 align="center">Predicting Airline Passenger Satisfaction with Machine Learning</h3>

<p align="center">
  <a href="https://skysense-viswanath.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
  </a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
</p>

<p align="center"><b>🚀 Live demo:</b> <a href="https://skysense-viswanath.streamlit.app/">skysense-viswanath.streamlit.app</a></p>

---

## What this is

I wanted to find out if a machine could read between the lines of a post-flight survey and tell us — before a passenger ever writes a complaint — whether they walked off the plane happy or annoyed.

**SkySense** does exactly that. You type a flight number, pick a route, fill in a few service ratings, and the app predicts whether the passenger will be **satisfied** or **neutral / dissatisfied**. Behind the curtain is a K-Nearest-Neighbours model trained on **129,880** real passenger surveys, wrapped in a Streamlit interface with a Three.js 3D result animation.

It's not just a notebook. It's a working product you can use right now.

---

## The numbers that matter

| Metric | Score |
|---|---|
| **Accuracy** | 92.9% |
| **Precision** | 94.9% |
| **Recall** | 88.5% |
| **F1-score** | 91.6% |

Evaluated on **25,976 unseen passengers** — a held-out 20% test set the model never touched during training.

---

## What makes it interesting

**It feels like a real flight booking.** Type `6E6199` and SkySense recognises it as an IndiGo flight. Pick **DEL → BOM** and the app instantly tells you it's a 707-mile journey — computed live from real airport coordinates using the haversine great-circle formula, no APIs, no internet calls. A boarding-pass card shows the carrier, flight number, route and distance — all auto-filled before you even click Predict.

**The ML pipeline is honest.** A lot of demo apps cheat by re-fitting their preprocessing on the input. SkySense doesn't. It bundles the trained model and every fitted encoder into a single `joblib` dict and only ever *transforms* new data with them — never re-fits. The result: predictions in the deployed app match the training notebook *exactly*. (I validated this against 500 training rows: zero mismatches.)

**The result is dramatic.** When a passenger is predicted satisfied, a Three.js particle field of 230 tiny 3D shapes rises in celebration (🥳). When dissatisfied, they fall (😞). It's just a visualisation — but it makes the app fun to demo and easy to remember.

---

## The model in one diagram
Raw input (22 features)
│
▼
┌────────────────────────────────────────┐
│  Gender, Customer Type, Type of Travel │ → OneHotEncoder    → 6 columns
│  Class                                 │ → OrdinalEncoder   → 1 column
│  Age, Distance, 14 ratings, 2 delays   │ → StandardScaler   → 18 columns
└────────────────────────────────────────┘
│
▼
25-feature vector (exact model order)
│
▼
KNeighborsClassifier (K=5)
│
▼
satisfied  |  neutral / dissatisfied  +  confidence %

---

## A design decision worth talking about

A flight number alone — like `6E6199` — can't tell you its route or distance unless you query a live airline schedule API (paid, requires keys, breaks the moment you're offline). I didn't want a demo that could die mid-presentation, so I split the problem:

| Input | What it tells the app | How |
|---|---|---|
| **Flight number** | The carrier (IndiGo, Emirates, …) | Parses the 2-letter IATA airline prefix |
| **Origin + Destination** | The flight distance | Haversine formula on real airport coordinates |

This way every part of the input does something real, no API keys are needed, and the app works offline — perfect for a presentation, an exam demo, or a recruiter on a flaky Wi-Fi.

---

## Running it locally

```bash
git clone https://github.com/viswanath-0/skysense.git
cd skysense
pip install -r requirements.txt
streamlit run app1.py
```

Opens at `http://localhost:8501`.

> Requires Python 3.12 and `model_prod1323_file.pkl` in the project root.

---

## Project layout
skysense/
├── app1.py                     # Streamlit app + boarding pass + 3D result scene
├── airports.py                 # Offline airport DB, airline codes, haversine engine
├── model_prod1323_file.pkl     # Trained model + all encoders, bundled
├── requirements.txt
├── .python-version             # Pins Python 3.12 for sklearn compatibility
└── README.md

---

## Tech I used

- **scikit-learn** — KNN + OneHotEncoder + OrdinalEncoder + StandardScaler + LabelEncoder
- **pandas / numpy** — data wrangling
- **Streamlit** — the web framework
- **Three.js** — the 3D particle result scene (embedded via `components.html`)
- **joblib** — serialising the model + encoders as one portable artifact
- **Python stdlib `math`** — the haversine implementation; no external geo dependencies

---

## Where it could go next

- Swap the offline carrier lookup for a live flight-schedule API to support arbitrary flight numbers
- Compare KNN against Random Forest / XGBoost — see if a tree-based model beats 92.9%
- Add SHAP value explanations so users see *why* the model made each prediction
- Expand the airport database beyond the current ~60 hubs

---

## About me

Built by **Suddapalli Viswanath** — currently learning data science and machine learning, one project at a time. SkySense is my first end-to-end ML project: data cleaning → preprocessing → modelling → deployment → live web app.

If you have feedback or just want to say hi, open an issue or [connect with me on GitHub](https://github.com/viswanath-0). I'd love to hear what you think.

---

<p align="center">
  <i>Built with Streamlit · scikit-learn · Three.js · and a lot of coffee ☕</i>
</p>


## 🎬 Demo video

🎥 **[Watch the SkySense demo on Google Drive](https://drive.google.com/file/d/1T0rNERnbf1evSnCWI_Xc79TQNAasrgC0/view?usp=sharing)**

> Demonstrates flight number parsing, auto-distance calculation, prediction, and the 3D result animation.
