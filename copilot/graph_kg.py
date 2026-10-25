"""
Optional Neo4j knowledge graph linking Engine -> Sensor -> FailureMode.

Entirely optional: if NEO4J_URI is unset or the driver/DB is unavailable, every
function falls back to the built-in map below, so the copilot runs without it.

Schema:
  (:Engine {id})-[:HAS_SENSOR]->(:Sensor {name})
  (:Sensor)-[:INDICATES]->(:FailureMode {name, mitigation})
"""
import os

NEO4J_URI = os.environ.get("NEO4J_URI")          # e.g. bolt://localhost:7687
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD")    # required when NEO4J_URI is set

# Sensor -> failure mode mapping (mirrors the manual's "Common Failure Modes").
SENSOR_FAILURE_MAP = {
    "HPC_outlet_temperature": ("HPC Degradation", "Water-wash for fouling; blade/clearance inspection."),
    "fuel_flow": ("HPC Degradation", "Efficiency check; investigate hot-section wear."),
    "HPC_outlet_pressure": ("Seal Leakage", "Seal inspection and replacement."),
    "fan_speed": ("Fan Module Degradation", "Balance check; blade dressing or replacement."),
    "vibration": ("Bearing Wear", "Oil-debris monitoring; bearing replacement at threshold."),
    "bearing_temperature": ("Bearing Wear", "Oil-debris analysis; bearing replacement."),
    "static_pressure": ("Seal Leakage", "Gas-path seal inspection."),
}

# C-MAPSS sensor columns -> the physical names above (C-MAPSS readme / Saxena 2008).
# vibration and bearing_temperature are not measured in C-MAPSS.
CMAPSS_ALIASES = {
    "sensor_3": "HPC_outlet_temperature",   # T30
    "sensor_7": "HPC_outlet_pressure",      # P30
    "sensor_8": "fan_speed",                # Nf, physical fan speed
    "sensor_13": "fan_speed",               # NRf, corrected fan speed
    "sensor_11": "static_pressure",         # Ps30, HPC outlet static pressure
    "sensor_12": "fuel_flow",               # phi, fuel flow / Ps30
}

VALID_SENSOR_NAMES = sorted(SENSOR_FAILURE_MAP) + sorted(CMAPSS_ALIASES, key=lambda s: int(s[7:]))

# Labels this module owns; seed() only ever deletes nodes with these labels.
_LABELS = ("Engine", "Sensor", "FailureMode")


def normalize_sensor(name):
    """Map user/LLM input like 'Fan Speed', 'fan-speed' or 'sensor_8' to a key
    of SENSOR_FAILURE_MAP. Returns None if it is not a known sensor."""
    if not isinstance(name, str):
        return None
    key = name.strip().replace("-", "_").replace(" ", "_")
    if key.lower() in CMAPSS_ALIASES:
        return CMAPSS_ALIASES[key.lower()]
    by_lower = {k.lower(): k for k in SENSOR_FAILURE_MAP}
    return by_lower.get(key.lower())


def _driver():
    if not NEO4J_URI:
        return None
    if not NEO4J_PASS:
        print("[kg] NEO4J_URI is set but NEO4J_PASSWORD is not; using the built-in map")
        return None
    try:
        from neo4j import GraphDatabase
        return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    except Exception as e:
        print(f"[kg] Neo4j unavailable: {e}")
        return None


def available():
    drv = _driver()
    if not drv:
        return False
    try:
        with drv.session() as s:
            s.run("RETURN 1")
        return True
    except Exception as e:
        print(f"[kg] Neo4j connection failed: {e}")
        return False
    finally:
        drv.close()


def seed(engine_ids=("TURBOFAN_001", "TURBOFAN_002", "TURBOFAN_003")):
    """(Re)create the Engine->Sensor->FailureMode graph. Only nodes with this
    module's labels are removed, so other data in the database is untouched.
    No-op if Neo4j is absent."""
    drv = _driver()
    if not drv:
        print("[kg] skipping seed (no Neo4j)")
        return False
    try:
        with drv.session() as s:
            for label in _LABELS:
                s.run(f"MATCH (n:{label}) DETACH DELETE n")
            for sensor, (mode, mitig) in SENSOR_FAILURE_MAP.items():
                s.run(
                    "MERGE (fm:FailureMode {name:$mode}) SET fm.mitigation=$mitig "
                    "MERGE (se:Sensor {name:$sensor}) "
                    "MERGE (se)-[:INDICATES]->(fm)",
                    mode=mode, mitig=mitig, sensor=sensor)
            for eid in engine_ids:
                s.run("MERGE (:Engine {id:$eid})", eid=eid)
                for sensor in SENSOR_FAILURE_MAP:
                    s.run(
                        "MATCH (e:Engine {id:$eid}),(se:Sensor {name:$sensor}) "
                        "MERGE (e)-[:HAS_SENSOR]->(se)", eid=eid, sensor=sensor)
        print("[kg] seeded graph")
        return True
    finally:
        drv.close()


def _static(sensor):
    m = SENSOR_FAILURE_MAP.get(sensor)
    return [{"mode": m[0], "mitigation": m[1]}] if m else []


def failure_modes_for_sensor(sensor_name):
    """Return [{mode, mitigation}] linked to a sensor. Falls back to the static
    map if Neo4j is unavailable, so the tool is always useful."""
    sensor = normalize_sensor(sensor_name)
    if sensor is None:
        return []
    drv = _driver()
    if not drv:
        return _static(sensor)
    try:
        with drv.session() as s:
            rows = s.run(
                "MATCH (:Sensor {name:$n})-[:INDICATES]->(fm:FailureMode) "
                "RETURN fm.name AS mode, fm.mitigation AS mitigation", n=sensor)
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"[kg] query failed: {e}")
        return _static(sensor)
    finally:
        drv.close()


if __name__ == "__main__":
    print("Neo4j available:", available())
    seed()
    print(failure_modes_for_sensor("sensor_3"))
