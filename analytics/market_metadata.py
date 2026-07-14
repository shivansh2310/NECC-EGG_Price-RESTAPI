from __future__ import annotations

import pandas as pd


# Geographic coordinates and indicative population weights are application
# metadata, never price data. Weights are deliberately exposed as a proxy and
# should be replaced when an authoritative production/consumption series exists.
MARKETS = {
    "Ahmedabad": (23.0225, 72.5714, 6.4),
    "Ajmer": (26.4499, 74.6399, 0.55),
    "Allahabad (CC)": (25.4358, 81.8463, 1.5),
    "Asansole": (23.6739, 86.9524, 1.24),
    "Barwala": (29.3675, 75.9081, 0.05),
    "Bengaluru (CC)": (12.9716, 77.5946, 8.5),
    "Bhopal": (23.2599, 77.4126, 1.8),
    "Brahmapur (OD)": (19.3149, 84.7941, 0.36),
    "Burdwan (CC)": (23.2324, 87.8615, 0.35),
    "Chennai (CC)": (13.0827, 80.2707, 8.7),
    "Chittoor": (13.2172, 79.1003, 0.18),
    "Delhi (CC)": (28.6139, 77.2090, 16.3),
    "E.Godavari": (16.9891, 82.2475, 5.15),
    "Hospet": (15.2689, 76.3909, 0.21),
    "Hyderabad": (17.3850, 78.4867, 7.75),
    "Indore (CC)": (22.7196, 75.8577, 2.2),
    "Jabalpur": (23.1815, 79.9864, 1.27),
    "Kanpur (CC)": (26.4499, 80.3319, 2.8),
    "Kolkata (WB)": (22.5726, 88.3639, 14.1),
    "Ludhiana": (30.9010, 75.8573, 1.62),
    "Luknow (CC)": (26.8467, 80.9462, 3.7),
    "Midnapur (KOL)": (22.4257, 87.3199, 0.17),
    "Mumbai (CC)": (19.0760, 72.8777, 18.4),
    "Muzaffurpur (CC)": (26.1209, 85.3647, 0.4),
    "Mysuru": (12.2958, 76.6394, 0.99),
    "Nagpur": (21.1458, 79.0882, 2.5),
    "Namakkal": (11.2194, 78.1677, 0.06),
    "Patna": (25.6093, 85.1376, 2.0),
    "Pune": (18.5204, 73.8567, 5.05),
    "Raipur": (21.2514, 81.6296, 1.12),
    "Ranchi  (CC)": (23.3441, 85.3096, 1.3),
    "Surat": (21.1702, 72.8311, 4.59),
    "Varanasi (CC)": (25.3176, 82.9739, 1.2),
    "Vijayawada": (16.5062, 80.6480, 1.49),
    "Vizag": (17.6868, 83.2185, 1.73),
    "W.Godavari": (16.7107, 81.0952, 3.94),
    "Warangal": (17.9689, 79.5941, 0.76),
}


def market_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"market": market, "latitude": values[0], "longitude": values[1], "population_weight": values[2]}
            for market, values in MARKETS.items()
        ]
    )

