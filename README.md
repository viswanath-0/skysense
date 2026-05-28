# SkySense ✈️ — Airline Passenger Satisfaction Predictor

A real-world web app that predicts whether an airline passenger will be **satisfied** or **neutral / dissatisfied**. You enter a **flight number** and a **route**; the app identifies the **carrier**, auto-computes the **flight distance** using the great-circle (haversine) formula from real airport coordinates, and feeds everything into a K-Nearest-Neighbours model with a full scikit-learn preprocessing pipeline. Results render in a live **Three.js 3D scene**.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)
![scikit--learn](https://img.shields.io/badge/scikit--learn-KNN-F7931E)

## ✨ Features

- **Flight-number aware** — type `6E6199` and the app recognises the carrier (IndiGo) from the airline prefix.
- **Automatic distance** — pick origin & destination airports and the flight distance is computed live via the haversine great-circle formula (validated to within ~1% of published airport-pair distances). No API keys, fully offline.
- **Boarding-pass UI** — a clean From → To card showing carrier, flight number and computed distance.
- **Faithful ML pipeline** — rebuilds the exact training-time transforms (OneHotEncoder → OrdinalEncoder → StandardScaler) in the precise 25-column order the model expects, so predictions match the training notebook exactly.
- **3D result scene** — a Three.js particle field rises in celebration (🥳) when a passenger is predicted satisfied, and falls (😞) when dissatisfied, with an animated confidence counter.

## 🛫 How the flight inputs work (honest design note)

A flight number on its own cannot reveal a route or distance without a live airline-schedule API. So SkySense splits the job into what is genuinely knowable offline:

| Input | What it gives you | How |
|---|---|---|
| Flight number (`6E6199`) | The **carrier** | Parses the 2-char IATA airline prefix |
| Origin + Destination | The **distance** | Haversine great-circle between airport coordinates |

This keeps the app accurate and demo-proof — it never depends on a network call.

## 📊 The Model

| Stage | Tool | Columns |
|---|---|---|
| Nominal encoding | `OneHotEncoder` | Gender, Customer Type, Type of Travel |
| Ordinal encoding | `OrdinalEncoder` | Class |
| Scaling | `StandardScaler` | 18 numeric (Age, Flight Distance, 14 service ratings, 2 delays) |
| Target encoding | `LabelEncoder` | satisfaction |
| Classifier | `KNeighborsClassifier` | — |

Trained on **129,880** passenger records (Kaggle: *Airline Passenger Satisfaction*).

## 🚀 Run locally

```bash
git clone https://github.com/<your-username>/skysense.git
cd skysense
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## 📁 Project structure

```
skysense/
├── app.py                      # Streamlit app + boarding pass + 3D result scene
├── airports.py                 # Offline airport DB, airline codes, haversine engine
├── model_prod1323_file.pkl     # Model + all encoders (joblib dict)
├── requirements.txt
└── README.md
```

## 🧠 Prediction flow

The `.pkl` bundles the model and every fitted encoder in one dict:

```python
{'model': KNeighborsClassifier, 'cat_encod': OneHotEncoder,
 'cato_encod': OrdinalEncoder, 'output_encod': LabelEncoder, 'num_encod': StandardScaler}
```

For each prediction the app one-hot-encodes the 3 nominal columns, ordinal-encodes `Class`, standardises the 18 numerics, concatenates them into the model's exact 25-feature order, and runs `predict` + `predict_proba`. Encoders are only ever *transformed* with (never re-fit), which prevents data leakage and keeps output identical to the notebook.

## 🧩 Extending it

- **Live flight lookup:** swap the offline carrier/route logic for a flight API (e.g. AviationStack, FlightAware) to resolve real flight numbers to live routes. Keep the haversine fallback for offline demos.
- **More airports:** add rows to `AIRPORTS` in `airports.py` (`IATA: (city, country, lat, lon)`).

## 📝 Notes

- If unpickling raises a scikit-learn version warning, install the same scikit-learn version you trained with to silence it.
- Routes longer than the model's training range (~4,983 mi) still predict, but are flagged as extrapolation in the UI.

---

Built with Streamlit · Three.js · scikit-learn · haversine distance.
