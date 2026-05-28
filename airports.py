"""
airports.py — offline airport + airline reference data and a great-circle
distance engine. No network calls, so it works anywhere (demos, planes, exams).
"""
from math import radians, sin, cos, asin, sqrt

# IATA -> (City name, Country, latitude, longitude)
AIRPORTS = {
    # ---- India ----
    "DEL": ("Delhi (Indira Gandhi Intl)", "India", 28.5562, 77.1000),
    "BOM": ("Mumbai (Chhatrapati Shivaji)", "India", 19.0896, 72.8656),
    "BLR": ("Bengaluru (Kempegowda)", "India", 13.1986, 77.7066),
    "MAA": ("Chennai Intl", "India", 12.9941, 80.1709),
    "HYD": ("Hyderabad (Rajiv Gandhi)", "India", 17.2403, 78.4294),
    "CCU": ("Kolkata (Netaji Subhas)", "India", 22.6547, 88.4467),
    "COK": ("Kochi Intl", "India", 10.1520, 76.4019),
    "GOI": ("Goa (Dabolim)", "India", 15.3808, 73.8314),
    "PNQ": ("Pune", "India", 18.5821, 73.9197),
    "AMD": ("Ahmedabad (Sardar Patel)", "India", 23.0772, 72.6347),
    "JAI": ("Jaipur Intl", "India", 26.8242, 75.8122),
    "LKO": ("Lucknow (Chaudhary Charan)", "India", 26.7606, 80.8893),
    "GAU": ("Guwahati (Borjhar)", "India", 26.1061, 91.5859),
    "IXC": ("Chandigarh", "India", 30.6735, 76.7885),
    "TRV": ("Thiruvananthapuram Intl", "India", 8.4821, 76.9200),
    "NAG": ("Nagpur (Dr. Ambedkar)", "India", 21.0922, 79.0472),
    "BBI": ("Bhubaneswar (Biju Patnaik)", "India", 20.2444, 85.8178),
    "VTZ": ("Visakhapatnam", "India", 17.7211, 83.2245),
    "IXB": ("Bagdogra", "India", 26.6812, 88.3286),
    "SXR": ("Srinagar", "India", 33.9871, 74.7742),
    # ---- North America ----
    "JFK": ("New York (JFK)", "USA", 40.6413, -73.7781),
    "EWR": ("Newark Liberty", "USA", 40.6895, -74.1745),
    "LGA": ("New York (LaGuardia)", "USA", 40.7769, -73.8740),
    "LAX": ("Los Angeles Intl", "USA", 33.9416, -118.4085),
    "SFO": ("San Francisco Intl", "USA", 37.6213, -122.3790),
    "ORD": ("Chicago O'Hare", "USA", 41.9742, -87.9073),
    "ATL": ("Atlanta (Hartsfield)", "USA", 33.6407, -84.4277),
    "DFW": ("Dallas/Fort Worth", "USA", 32.8998, -97.0403),
    "SEA": ("Seattle-Tacoma", "USA", 47.4502, -122.3088),
    "BOS": ("Boston Logan", "USA", 42.3656, -71.0096),
    "IAD": ("Washington Dulles", "USA", 38.9531, -77.4565),
    "MIA": ("Miami Intl", "USA", 25.7959, -80.2871),
    "YYZ": ("Toronto Pearson", "Canada", 43.6777, -79.6248),
    "YVR": ("Vancouver Intl", "Canada", 49.1967, -123.1815),
    "MEX": ("Mexico City Intl", "Mexico", 19.4361, -99.0719),
    # ---- Europe ----
    "LHR": ("London Heathrow", "UK", 51.4700, -0.4543),
    "LGW": ("London Gatwick", "UK", 51.1537, -0.1821),
    "CDG": ("Paris Charles de Gaulle", "France", 49.0097, 2.5479),
    "FRA": ("Frankfurt Intl", "Germany", 50.0379, 8.5622),
    "MUC": ("Munich Intl", "Germany", 48.3538, 11.7861),
    "AMS": ("Amsterdam Schiphol", "Netherlands", 52.3105, 4.7683),
    "MAD": ("Madrid Barajas", "Spain", 40.4983, -3.5676),
    "BCN": ("Barcelona El Prat", "Spain", 41.2974, 2.0833),
    "FCO": ("Rome Fiumicino", "Italy", 41.8003, 12.2389),
    "ZRH": ("Zurich Intl", "Switzerland", 47.4647, 8.5492),
    "IST": ("Istanbul Intl", "Turkey", 41.2753, 28.7519),
    "SVO": ("Moscow Sheremetyevo", "Russia", 55.9726, 37.4146),
    # ---- Middle East ----
    "DXB": ("Dubai Intl", "UAE", 25.2532, 55.3657),
    "AUH": ("Abu Dhabi Intl", "UAE", 24.4330, 54.6511),
    "DOH": ("Doha (Hamad Intl)", "Qatar", 25.2731, 51.6080),
    # ---- Asia-Pacific ----
    "SIN": ("Singapore Changi", "Singapore", 1.3644, 103.9915),
    "HKG": ("Hong Kong Intl", "Hong Kong", 22.3080, 113.9185),
    "BKK": ("Bangkok Suvarnabhumi", "Thailand", 13.6900, 100.7501),
    "KUL": ("Kuala Lumpur Intl", "Malaysia", 2.7456, 101.7099),
    "NRT": ("Tokyo Narita", "Japan", 35.7720, 140.3929),
    "HND": ("Tokyo Haneda", "Japan", 35.5494, 139.7798),
    "ICN": ("Seoul Incheon", "South Korea", 37.4602, 126.4407),
    "PEK": ("Beijing Capital", "China", 40.0799, 116.6031),
    "PVG": ("Shanghai Pudong", "China", 31.1443, 121.8083),
    "SYD": ("Sydney Kingsford Smith", "Australia", -33.9399, 151.1753),
    "MEL": ("Melbourne Tullamarine", "Australia", -37.6690, 144.8410),
    # ---- Other ----
    "GRU": ("Sao Paulo Guarulhos", "Brazil", -23.4356, -46.4731),
    "JNB": ("Johannesburg (OR Tambo)", "South Africa", -26.1392, 28.2460),
}

# 2-char airline prefix -> carrier name
AIRLINES = {
    "6E": "IndiGo", "AI": "Air India", "UK": "Vistara", "SG": "SpiceJet",
    "G8": "Go First", "I5": "AIX Connect", "QP": "Akasa Air", "9I": "Alliance Air",
    "EK": "Emirates", "EY": "Etihad Airways", "QR": "Qatar Airways",
    "SQ": "Singapore Airlines", "CX": "Cathay Pacific", "TG": "Thai Airways",
    "MH": "Malaysia Airlines", "BA": "British Airways", "VS": "Virgin Atlantic",
    "LH": "Lufthansa", "AF": "Air France", "KL": "KLM", "TK": "Turkish Airlines",
    "AA": "American Airlines", "DL": "Delta Air Lines", "UA": "United Airlines",
    "AC": "Air Canada", "QF": "Qantas", "NH": "ANA", "JL": "Japan Airlines",
    "KE": "Korean Air", "CA": "Air China", "MU": "China Eastern",
}


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points, in statute miles."""
    R = 3958.7613  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def distance_between(origin_iata, dest_iata):
    """Miles between two airports (rounded int). None if same/unknown."""
    if origin_iata == dest_iata:
        return None
    o, d = AIRPORTS.get(origin_iata), AIRPORTS.get(dest_iata)
    if not o or not d:
        return None
    return int(round(haversine_miles(o[2], o[3], d[2], d[3])))


def parse_flight_number(fn: str):
    """'6E6199' -> ('6E', 'IndiGo', '6199'). Returns (code, airline, digits)."""
    fn = (fn or "").strip().upper().replace(" ", "")
    if len(fn) < 3:
        return None, "Unknown carrier", ""
    code = fn[:2]
    digits = fn[2:]
    return code, AIRLINES.get(code, "Unknown carrier"), digits


def airport_options():
    """Display strings for selectboxes, sorted: India first, then alphabetical."""
    def keyer(item):
        iata, (city, country, *_rest) = item
        return (0 if country == "India" else 1, country, city)
    out = []
    for iata, (city, country, *_rest) in sorted(AIRPORTS.items(), key=keyer):
        out.append(f"{iata} — {city}, {country}")
    return out


def iata_from_option(option: str) -> str:
    return option.split(" — ")[0]
