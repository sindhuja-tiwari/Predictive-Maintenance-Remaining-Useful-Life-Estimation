"""
Optional Neo4j knowledge graph linking Engine -> Sensor -> FailureMode.

Entirely optional: if NEO4J_URI is unset or the driver/DB is unavailable, every
function degrades to a no-op / empty result so the copilot still runs without it.

Schema:
  (:Engine {id})-[:HAS_SENSOR]->(:Sensor {name})
  (:Sensor)-[:INDICATES]->(:FailureMode {name, mitigation})
"""
import os

NEO4J_URI = os.environ.get("NEO4J_URI")          # e.g. bolt://localhost:7687
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD", "password")

# Sensor -> failure mode mapping (mirrors the manual's Section 6).
SENSOR_FAILURE_MAP = {
    "HPC_outlet_temperature": ("HPC Degradation", "Water-wash for fouling; blade/clearance inspection."),
    "fuel_flow": ("HPC Degradation", "Efficiency check; investigate hot-section wear."),
    "HPC_outlet_pressure": ("Seal Leakage", "Seal inspection and replacement."),
    "fan_speed": ("Fan Module Degradation", "Balance check; blade dressing or replacement."),
    "vibration": ("Bearing Wear", "Oil-debris monitoring; bearing replacement at threshold."),
    "bearing_temperature": ("Bearing Wear", "Oil-debris analysis; bearing replacement."),
    "static_pressure": ("Seal Leakage", "Gas-path seal inspection."),
}


def _driver():
    if not NEO4J_URI:
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
    """Create the Engine->Sensor->FailureMode graph. No-op if Neo4j is absent."""
    drv = _driver()
    if not drv:
        print("[kg] skipping seed (no Neo4j)")
        return False
    try:
        with drv.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
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


def failure_modes_for_sensor(sensor_name):
    """Return [{mode, mitigation}] linked to a sensor. Falls back to the static
    map if Neo4j is unavailable, so the tool is always useful."""
    drv = _driver()
    if not drv:
        m = SENSOR_FAILURE_MAP.get(sensor_name)
        return [{"mode": m[0], "mitigation": m[1]}] if m else []
    try:
        with drv.session() as s:
            rows = s.run(
                "MATCH (:Sensor {name:$n})-[:INDICATES]->(fm:FailureMode) "
                "RETURN fm.name AS mode, fm.mitigation AS mitigation", n=sensor_name)
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"[kg] query failed: {e}")
        m = SENSOR_FAILURE_MAP.get(sensor_name)
        return [{"mode": m[0], "mitigation": m[1]}] if m else []
    finally:
        drv.close()


if __name__ == "__main__":
    print("Neo4j available:", available())
    seed()
    print(failure_modes_for_sensor("vibration"))