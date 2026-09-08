#!/bin/bash

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

python - <<'PY'
import ast
import json
from pathlib import Path

python_files = [
    path for path in Path(".").rglob("*.py")
    if ".git" not in path.parts
]
for path in python_files:
    ast.parse(path.read_text(), filename=str(path))

json_files = [
    path for path in Path(".").rglob("*.json")
    if ".git" not in path.parts
]
for path in json_files:
    with path.open() as handle:
        json.load(handle)

from relnet.evaluation.experiment_profiles import PROFILES, load_profile

for profile in PROFILES:
    config = load_profile(profile)
    names = [item['name'] for item in config['experiments']]
    if not names or len(names) != len(set(names)):
        raise SystemExit('Invalid evaluation profile: ' + profile)

required = [
    Path('run_preference_training.py'),
    Path('relnet/agent/rnet_dqn/preference_agent.py'),
    Path('tools/calibrate_normalization.py'),
    Path('myriad/celery/run_celery_job.sh'),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit('Missing release files: ' + ', '.join(missing))

for path in python_files + json_files:
    text = path.read_text()
    if 'experiments/preference_conditioned' in text:
        raise SystemExit('Obsolete preference path in ' + str(path))

q_net = ast.parse(Path('relnet/agent/rnet_dqn/q_net.py').read_text())
classes = {
    node.name: [getattr(base, 'id', None) for base in node.bases]
    for node in q_net.body if isinstance(node, ast.ClassDef)
}
if classes.get('PreferenceQNet') != ['QNet']:
    raise SystemExit('PreferenceQNet must extend the original QNet')

agent = ast.parse(
    Path('relnet/agent/rnet_dqn/preference_agent.py').read_text())
classes = {
    node.name: [getattr(base, 'id', None) for base in node.bases]
    for node in agent.body if isinstance(node, ast.ClassDef)
}
if classes.get('PreferenceConditionedRNetDQNAgent') != ['RNetDQNAgent']:
    raise SystemExit('Preference agent must extend RNetDQNAgent')

print("Parsed {} Python and {} JSON files.".format(
    len(python_files), len(json_files)
))
PY

while IFS= read -r -d '' script
do
  bash -n "$script"
done < <(find . -type f -name '*.sh' -print0)

printf 'Release validation passed.\n'
