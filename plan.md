## Implementation plan (starting point)

This repo already has a working Streamlit UI (`app.py`) for:

- Crop yield prediction (RF + XGBoost) using manual inputs
- Plant disease classification from leaf images (Keras `.keras` models)

The missing “hackathon wow” feature from `docs/List of Features.md` is **live weather API integration** so farmers don’t manually enter temperature/rain/humidity/etc.

---

## External APIs we will use (real endpoints + docs)

### Open‑Meteo Geocoding API (district/state → lat/long)

- **Docs**: `https://open-meteo.com/en/docs/geocoding-api`
- **Endpoint**: `GET https://geocoding-api.open-meteo.com/v1/search`
- **Query params (used)**:
  - `name`: free-text place name (e.g. `"Ahmedabad, Gujarat, India"`)
  - `count`: number of candidates (we’ll use `1`)
  - `language`: e.g. `en`
  - `countryCode`: `IN` (to reduce ambiguity)
- **Response (key fields)**:
  - `results[0].latitude`, `results[0].longitude`
  - `results[0].name`, `results[0].admin1`, `results[0].country`

### Open‑Meteo Forecast API (lat/long → weather variables)

- **Docs**: `https://open-meteo.com/en/docs`
- **Endpoint**: `GET https://api.open-meteo.com/v1/forecast`
- **Query params (used)**:
  - `latitude`, `longitude`
  - `hourly`: `temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,shortwave_radiation`
  - `timezone`: `auto`
  - `forecast_days`: `1` (today)
- **Response (key fields)**:
  - `hourly.time[]` and `hourly.<var>[]` arrays (same length)

**Mapping to model inputs**

- `Temperature_C` ← `temperature_2m` (°C)
- `Humidity_%` ← `relative_humidity_2m` (%)
- `Rainfall_mm` ← `precipitation` (mm)
- `Wind_Speed_m_s` ← `wind_speed_10m` (m/s)
- `Solar_Radiation_MJ_m2_day` ← `shortwave_radiation` (W/m² hourly; we’ll approximate by using “current hour” value as a proxy until we implement proper daily integration)

### SoilGrids REST API (lat/long → soil properties)

- **Docs (Swagger)**: `https://rest.isric.org/soilgrids/v2.0/docs`
- **Docs (overview)**: `https://docs.isric.org/globaldata/soilgrids/`
- **Endpoint**: `GET https://rest.isric.org/soilgrids/v2.0/properties/query`
- **Query params (used)**:
  - `lat`, `lon`
  - `property`: e.g. `phh2o,nitrogen,soc,clay,sand,silt`
  - `depth`: `0-5cm`
  - `value`: `mean`
- **Notes**:
  - SoilGrids can be **rate-limited / temporarily down** (fair-use + beta service), so the app must fall back to manual inputs.
  - `phh2o` is commonly stored as **pH×10**; divide by 10 when values look like 50–80.

---

## Work items to implement now

### 1) Streamlit: “Auto-fill climate from live weather”

- Add a sidebar or form toggle to fetch weather
- Geocode `"district, state, India"` → lat/long
- Fetch Open‑Meteo hourly forecast for today
- Fill Streamlit inputs using `st.session_state` (so sliders update)
- Fail gracefully (network/offline → keep manual sliders)

### 2) Dependencies

- Add `requests` to `requirements.txt` (and a minimal list of app deps)

### 3) Docs

- Keep this `plan.md` as the single source of truth for endpoints + parameters.
