"""
SkySense — Airline Passenger Satisfaction Predictor
A real-world Streamlit app: enter a flight number + route, the app identifies the
carrier and auto-computes the flight distance (great-circle / haversine), then a
KNN model + full sklearn pipeline predicts passenger satisfaction.

Run locally with:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from airports import (
    AIRPORTS,
    distance_between,
    parse_flight_number,
    airport_options,
    iata_from_option,
)

# --------------------------------------------------------------------------- #
#  PAGE CONFIG
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="SkySense · Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "model_prod1323_file.pkl"


# --------------------------------------------------------------------------- #
#  LOAD THE PRODUCTION BUNDLE (model + all encoders) — cached so it loads once
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_bundle(path: str):
    bundle = joblib.load(path)
    return (
        bundle["model"],        # KNeighborsClassifier
        bundle["cat_encod"],    # OneHotEncoder  -> Gender, Customer Type, Type of Travel
        bundle["cato_encod"],   # OrdinalEncoder -> Class
        bundle["output_encod"], # LabelEncoder   -> satisfaction
        bundle["num_encod"],    # StandardScaler -> 18 numeric columns
    )


try:
    MODEL, OHE, OE, LE, STD = load_bundle(MODEL_PATH)
    NUM_COLS = list(STD.feature_names_in_)
    FINAL_ORDER = list(MODEL.feature_names_in_)
    MODEL_OK = True
except Exception as e:                               # noqa: BLE001
    MODEL_OK = False
    LOAD_ERR = str(e)


# --------------------------------------------------------------------------- #
#  PREDICTION — rebuilds the EXACT training pipeline (validated: 0 mismatches)
# --------------------------------------------------------------------------- #
def predict(raw: dict):
    df = pd.DataFrame([raw])

    cat = OHE.transform(df[["Gender", "Customer Type", "Type of Travel"]])
    cat = pd.DataFrame(np.asarray(cat), columns=list(OHE.get_feature_names_out()))

    cato = OE.transform(df[["Class"]])
    cato = pd.DataFrame(np.asarray(cato), columns=["Class"])

    num = pd.DataFrame(STD.transform(df[NUM_COLS]), columns=NUM_COLS)

    X = pd.concat([cat, cato, num], axis=1)[FINAL_ORDER]

    code = MODEL.predict(X)[0]
    proba = MODEL.predict_proba(X)[0]
    label = LE.inverse_transform([code])[0]
    confidence = float(np.max(proba)) * 100.0
    return label, confidence, proba


# --------------------------------------------------------------------------- #
#  GLOBAL STYLING
# --------------------------------------------------------------------------- #
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@600;700&display=swap');

:root{ --neon:#5eead4; --neon2:#818cf8; --pink:#f472b6; --ink:#e2e8f0; --muted:#94a3b8; --card:rgba(255,255,255,.04);}
#MainMenu, footer, header {visibility:hidden;}

.stApp{
    background:
        radial-gradient(900px 600px at 12% -10%, rgba(129,140,248,.20), transparent 60%),
        radial-gradient(900px 600px at 90% 0%, rgba(94,234,212,.16), transparent 55%),
        radial-gradient(800px 700px at 50% 120%, rgba(244,114,182,.14), transparent 60%),
        #060b18;
    background-attachment:fixed; color:var(--ink); font-family:'Inter',sans-serif;
}
.stApp::before{
    content:""; position:fixed; inset:-50%; z-index:0; pointer-events:none;
    background:conic-gradient(from 0deg at 50% 50%,
        rgba(129,140,248,.07), rgba(94,234,212,.07), rgba(244,114,182,.07), rgba(129,140,248,.07));
    filter:blur(80px); animation:spin 28s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg);}}
.block-container{position:relative; z-index:1; padding-top:2rem; max-width:1180px;}

.hero{text-align:center; padding:1.6rem 0 .4rem;}
.hero h1{
    font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:3.2rem; line-height:1.05; margin:0;
    background:linear-gradient(100deg,#fff,var(--neon) 40%,var(--neon2) 70%,var(--pink));
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-1px;
}
.hero p{color:var(--muted); font-size:1.05rem; margin:.5rem 0 0; letter-spacing:.3px;}
.badge{display:inline-block; margin-top:.9rem; padding:.32rem .9rem; border-radius:999px;
    font-size:.78rem; font-weight:600; color:var(--neon);
    background:rgba(94,234,212,.08); border:1px solid rgba(94,234,212,.25);}

.sec{font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1.15rem;
    margin:1.4rem 0 .2rem; color:#fff; display:flex; align-items:center; gap:.5rem;}
.sec .dot{width:9px; height:9px; border-radius:50%; background:var(--neon); box-shadow:0 0 12px var(--neon);}

div[data-testid="stHorizontalBlock"]{
    background:var(--card); border:1px solid rgba(255,255,255,.07); border-radius:18px;
    padding:1.1rem 1.2rem; margin:.4rem 0 .2rem; backdrop-filter:blur(14px);
}
label{color:var(--ink)!important; font-weight:500!important;}
.stSlider [data-baseweb="slider"] div[role="slider"]{background:var(--neon)!important; box-shadow:0 0 10px var(--neon)!important;}

.stButton>button{
    width:100%; padding:.95rem 1rem; margin-top:1.4rem;
    font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.12rem;
    color:#04121a; border:none; border-radius:16px; cursor:pointer;
    background:linear-gradient(100deg,var(--neon),var(--neon2));
    box-shadow:0 10px 30px rgba(94,234,212,.30); transition:transform .15s ease, box-shadow .15s ease;
}
.stButton>button:hover{transform:translateY(-2px); box-shadow:0 16px 40px rgba(94,234,212,.45);}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
    <h1>SkySense ✈️</h1>
    <p>Enter a flight & route — we identify the carrier, compute the distance, and predict satisfaction</p>
    <span class="badge">KNN · scikit-learn · live great-circle distance · 129,880 records</span>
</div>
""",
    unsafe_allow_html=True,
)

if not MODEL_OK:
    st.error(f"Couldn't load **{MODEL_PATH}**. Keep it next to app.py.\n\nDetails: {LOAD_ERR}")
    st.stop()


# --------------------------------------------------------------------------- #
#  FLIGHT & ROUTE  (flight number -> carrier, route -> auto distance)
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec"><span class="dot"></span>Flight & route</div>', unsafe_allow_html=True)

opts = airport_options()
default_from = next((i for i, o in enumerate(opts) if o.startswith("DEL")), 0)
default_to = next((i for i, o in enumerate(opts) if o.startswith("BOM")), 1)

f1, f2, f3 = st.columns(3)
with f1:
    flight_no = st.text_input("Flight number", value="6E6199", max_chars=7)
with f2:
    origin_opt = st.selectbox("From (origin)", opts, index=default_from)
with f3:
    dest_opt = st.selectbox("To (destination)", opts, index=default_to)

code, airline, digits = parse_flight_number(flight_no)
origin = iata_from_option(origin_opt)
dest = iata_from_option(dest_opt)
auto_distance = distance_between(origin, dest)

# Boarding-pass style summary card
def boarding_pass(airline, code, digits, origin, dest, distance):
    o_city = AIRPORTS.get(origin, ("?", "", 0, 0))[0].split(" (")[0]
    d_city = AIRPORTS.get(dest, ("?", "", 0, 0))[0].split(" (")[0]
    dist_txt = f"{distance:,} mi" if distance else "— set a valid route —"
    html = """
    <div style="display:flex;align-items:stretch;gap:0;border-radius:16px;overflow:hidden;
         border:1px solid rgba(255,255,255,.10);background:linear-gradient(120deg,#0c1730,#0a1226);
         font-family:'Space Grotesk',sans-serif;color:#fff;margin:.2rem 0 .4rem;">
      <div style="flex:1;padding:18px 22px;">
        <div style="font-size:.72rem;letter-spacing:2px;color:#5eead4;text-transform:uppercase;">__AIRLINE__</div>
        <div style="display:flex;align-items:flex-end;gap:18px;margin-top:10px;">
          <div><div style="font-size:2.1rem;font-weight:700;line-height:1;">__O__</div>
               <div style="font-size:.72rem;color:#94a3b8;margin-top:4px;">__OCITY__</div></div>
          <div style="flex:1;text-align:center;color:#5eead4;font-size:1.3rem;margin-bottom:14px;">✈ — — —</div>
          <div style="text-align:right;"><div style="font-size:2.1rem;font-weight:700;line-height:1;">__D__</div>
               <div style="font-size:.72rem;color:#94a3b8;margin-top:4px;">__DCITY__</div></div>
        </div>
      </div>
      <div style="width:1px;background:repeating-linear-gradient(#ffffff33 0 6px,transparent 6px 12px);"></div>
      <div style="width:170px;padding:18px;display:flex;flex-direction:column;justify-content:center;
           background:rgba(94,234,212,.05);">
        <div style="font-size:.66rem;letter-spacing:1.5px;color:#7c8aa3;text-transform:uppercase;">Flight</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.25rem;font-weight:700;">__FN__</div>
        <div style="font-size:.66rem;letter-spacing:1.5px;color:#7c8aa3;text-transform:uppercase;margin-top:12px;">Distance</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.25rem;font-weight:700;color:#5eead4;">__DIST__</div>
      </div>
    </div>
    """
    repl = {
        "__AIRLINE__": airline, "__O__": origin, "__D__": dest,
        "__OCITY__": o_city, "__DCITY__": d_city,
        "__FN__": f"{code}{digits}" if digits else "—", "__DIST__": dist_txt,
    }
    for k, v in repl.items():
        html = html.replace(k, str(v))
    st.markdown(html, unsafe_allow_html=True)

boarding_pass(airline, code, digits, origin, dest, auto_distance)

if auto_distance is None:
    st.warning("Origin and destination must be different airports to compute a distance.")
elif auto_distance > 4983:
    st.info(
        f"Heads-up: this route is {auto_distance:,} mi — longer than the model's training "
        f"range (max ~4,983 mi). The prediction still runs, but treat ultra-long-haul as extrapolation."
    )

flight_distance = auto_distance if auto_distance else 850  # safe fallback for the model


# --------------------------------------------------------------------------- #
#  PASSENGER PROFILE
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec"><span class="dot"></span>Passenger profile</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 7, 85, 35)
with c2:
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
with c3:
    travel_class = st.selectbox("Class", ["Business", "Eco Plus", "Eco"])


# --------------------------------------------------------------------------- #
#  SERVICE RATINGS
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec"><span class="dot"></span>Service ratings (0 = N/A · 1 = poor · 5 = excellent)</div>', unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)
with s1:
    wifi = st.slider("Inflight wifi service", 0, 5, 3)
    booking = st.slider("Ease of Online booking", 0, 5, 3)
    food = st.slider("Food and drink", 0, 5, 3)
    seat = st.slider("Seat comfort", 0, 5, 3)
    onboard = st.slider("On-board service", 0, 5, 3)
with s2:
    time_conv = st.slider("Departure/Arrival time convenient", 0, 5, 3)
    gate = st.slider("Gate location", 0, 5, 3)
    boarding = st.slider("Online boarding", 0, 5, 3)
    entertainment = st.slider("Inflight entertainment", 0, 5, 3)
    legroom = st.slider("Leg room service", 0, 5, 3)
with s3:
    baggage = st.slider("Baggage handling", 1, 5, 3)
    checkin = st.slider("Checkin service", 0, 5, 3)
    inflight_svc = st.slider("Inflight service", 0, 5, 3)
    clean = st.slider("Cleanliness", 0, 5, 3)


# --------------------------------------------------------------------------- #
#  FLIGHT DELAYS
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec"><span class="dot"></span>Flight delays</div>', unsafe_allow_html=True)
d1, d2 = st.columns(2)
with d1:
    dep_delay = st.number_input("Departure Delay (minutes)", 0, 1600, 0, step=1)
with d2:
    arr_delay = st.number_input("Arrival Delay (minutes)", 0, 1600, 0, step=1)


# --------------------------------------------------------------------------- #
#  3D RESULT RENDERER — Three.js particle scene + floating 3D emoji
# --------------------------------------------------------------------------- #
def render_result(label, confidence, airline, origin, dest, distance):
    satisfied = label == "satisfied"
    emoji = "🥳" if satisfied else "😞"
    sub_emoji = "😄" if satisfied else "😤"
    headline = "SATISFIED" if satisfied else "NEUTRAL / DISSATISFIED"
    color = "#5eead4" if satisfied else "#fb7185"
    color2 = "#34d399" if satisfied else "#f43f5e"
    route = f"{airline} · {origin} → {dest} · {distance:,} mi" if distance else f"{airline} · {origin} → {dest}"
    msg = ("This passenger is predicted to walk away happy. 🌟" if satisfied
           else "This passenger is predicted to be unhappy with the experience.")

    template = r"""
<div id="stage">
  <canvas id="fx"></canvas>
  <div id="card">
    <div id="emoji">__EMOJI__</div>
    <div id="ring"></div>
    <div id="head">__HEAD__ <span id="se">__SUB__</span></div>
    <div id="route">__ROUTE__</div>
    <div id="msg">__MSG__</div>
    <div id="confwrap">
      <div id="conflabel">Model confidence</div>
      <div id="bar"><div id="fill"></div></div>
      <div id="confval">0%</div>
    </div>
  </div>
</div>
<style>
  #stage{position:relative;width:100%;height:470px;border-radius:22px;overflow:hidden;
    background:radial-gradient(700px 400px at 50% 0%, rgba(255,255,255,.05), transparent 60%), #070d1c;
    border:1px solid rgba(255,255,255,.08);font-family:'Inter',system-ui,sans-serif;}
  #fx{position:absolute;inset:0;width:100%;height:100%;}
  #card{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;z-index:2;}
  #emoji{font-size:118px;line-height:1;filter:drop-shadow(0 18px 30px rgba(0,0,0,.5));
    animation:floaty 3.2s ease-in-out infinite;transform-style:preserve-3d;}
  @keyframes floaty{0%,100%{transform:translateY(0) rotateY(-12deg) rotateZ(-2deg);}50%{transform:translateY(-22px) rotateY(12deg) rotateZ(2deg);}}
  #ring{position:absolute;top:58px;width:210px;height:210px;border-radius:50%;border:2px solid __COLOR__;opacity:.25;animation:pulse 2.6s ease-out infinite;}
  @keyframes pulse{0%{transform:scale(.7);opacity:.5;}100%{transform:scale(1.6);opacity:0;}}
  #head{margin-top:16px;font-family:'Space Grotesk','Inter',sans-serif;font-weight:700;font-size:1.95rem;letter-spacing:1px;color:#fff;text-shadow:0 0 24px __COLOR__;}
  #se{font-size:1.5rem;}
  #route{margin-top:8px;color:#5eead4;font-size:.85rem;letter-spacing:1.5px;text-transform:uppercase;}
  #msg{margin-top:.45rem;color:#9fb0c9;font-size:1rem;max-width:460px;padding:0 20px;}
  #confwrap{margin-top:20px;width:320px;}
  #conflabel{color:#7c8aa3;font-size:.78rem;text-transform:uppercase;letter-spacing:1.5px;}
  #bar{margin-top:8px;height:12px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;}
  #fill{height:100%;width:0%;border-radius:999px;background:linear-gradient(90deg,__COLOR__,__COLOR2__);box-shadow:0 0 18px __COLOR__;transition:width 1.4s cubic-bezier(.2,.8,.2,1);}
  #confval{margin-top:6px;font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.5rem;color:#fff;}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function(){
  var SAT = __SAT__;
  var COL = SAT ? [0x5eead4,0x34d399,0x818cf8,0xfde68a] : [0xfb7185,0xf43f5e,0x64748b,0x94a3b8];
  var canvas=document.getElementById('fx');
  var scene=new THREE.Scene();
  var camera=new THREE.PerspectiveCamera(60, canvas.clientWidth/canvas.clientHeight,0.1,1000);
  camera.position.z=60;
  var renderer=new THREE.WebGLRenderer({canvas:canvas,alpha:true,antialias:true});
  renderer.setPixelRatio(window.devicePixelRatio);
  function size(){renderer.setSize(canvas.clientWidth,canvas.clientHeight,false);
    camera.aspect=canvas.clientWidth/canvas.clientHeight;camera.updateProjectionMatrix();}
  size();window.addEventListener('resize',size);
  var N=230,parts=[];
  var geos=[new THREE.TetrahedronGeometry(1.1),new THREE.BoxGeometry(1.4,1.4,1.4),new THREE.OctahedronGeometry(1.2)];
  for(var i=0;i<N;i++){
    var g=geos[i%geos.length];
    var m=new THREE.MeshBasicMaterial({color:COL[i%COL.length],transparent:true,opacity:.85});
    var mesh=new THREE.Mesh(g,m);
    mesh.position.set((Math.random()-.5)*120,(Math.random()-.5)*90,(Math.random()-.5)*60);
    mesh.rotation.set(Math.random()*6,Math.random()*6,0);
    mesh.userData={vy:SAT?(0.12+Math.random()*0.28):-(0.10+Math.random()*0.22),
      vx:(Math.random()-.5)*0.10,rx:(Math.random()-.5)*0.04,ry:(Math.random()-.5)*0.04,s:0.5+Math.random()*1.2};
    mesh.scale.setScalar(mesh.userData.s);scene.add(mesh);parts.push(mesh);
  }
  function animate(){
    requestAnimationFrame(animate);
    for(var i=0;i<parts.length;i++){var p=parts[i];
      p.position.y+=p.userData.vy;p.position.x+=p.userData.vx;
      p.rotation.x+=p.userData.rx;p.rotation.y+=p.userData.ry;
      if(p.position.y>50)p.position.y=-50; if(p.position.y<-50)p.position.y=50;}
    scene.rotation.y+=0.0009;renderer.render(scene,camera);
  }
  animate();
  var target=__CONF__;
  setTimeout(function(){document.getElementById('fill').style.width=target.toFixed(1)+'%';},150);
  var cur=0,cv=document.getElementById('confval');
  var t=setInterval(function(){cur+=target/40;if(cur>=target){cur=target;clearInterval(t);}cv.textContent=cur.toFixed(1)+'%';},35);
})();
</script>
"""
    html = (template
        .replace("__EMOJI__", emoji).replace("__SUB__", sub_emoji)
        .replace("__HEAD__", headline).replace("__ROUTE__", route).replace("__MSG__", msg)
        .replace("__COLOR2__", color2).replace("__COLOR__", color)
        .replace("__SAT__", "true" if satisfied else "false")
        .replace("__CONF__", f"{confidence:.1f}"))
    components.html(html, height=490)


# --------------------------------------------------------------------------- #
#  PREDICT
# --------------------------------------------------------------------------- #
if st.button("🔮  Predict Satisfaction"):
    raw = {
        "Gender": gender, "Customer Type": customer_type, "Type of Travel": travel_type,
        "Class": travel_class, "Age": age, "Flight Distance": flight_distance,
        "Inflight wifi service": wifi, "Departure/Arrival time convenient": time_conv,
        "Ease of Online booking": booking, "Gate location": gate, "Food and drink": food,
        "Online boarding": boarding, "Seat comfort": seat, "Inflight entertainment": entertainment,
        "On-board service": onboard, "Leg room service": legroom, "Baggage handling": baggage,
        "Checkin service": checkin, "Inflight service": inflight_svc, "Cleanliness": clean,
        "Departure Delay in Minutes": dep_delay, "Arrival Delay in Minutes": arr_delay,
    }
    label, confidence, _ = predict(raw)
    render_result(label, confidence, airline, origin, dest, auto_distance)

st.markdown(
    "<p style='text-align:center;color:#5b6b86;font-size:.8rem;margin-top:2rem;'>"
    "Built with Streamlit · Three.js · scikit-learn · haversine distance — © SkySense</p>",
    unsafe_allow_html=True,
)
