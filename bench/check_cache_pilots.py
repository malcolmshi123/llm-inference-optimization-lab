import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

for name in ("pilot-off", "pilot-on"):
    run = json.loads((root / name / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed", f"{name}: {run['status']}"

    print(f"\n{name}: completed")
    for scenario in run["scenarios"]:
        metrics = scenario["metrics"]
        print(
            f"  {scenario['name']}: "
            f"queries={metrics.get('prefix_queries_tokens')}, "
            f"hits={metrics.get('prefix_hits_tokens')}, "
            f"hit_rate={metrics.get('token_cache_hit_rate')}"
        )