# Tasks and trajectories

Tasks describe **what to produce**. Trajectories describe **actions taken** to produce a
canvas. Global limits and scoring settings remain in `src/marketcanvas_env/config`.

```text
data/
├── tasks/                  # Five plain TaskSpec YAMLs
│   ├── summer_sale.yaml   # Summer Sale
│   ├── webinar.yaml
│   ├── newsletter.yaml
│   ├── event.yaml
│   └── two_column.yaml
└── trajectories/
    └── <task_name>/        # Three YAMLs for each task
        ├── well_done.yaml
        └── <two representative attacks>.yaml
```

Each trajectory contains `task` (the task filename stem), `description`, `seed`, and an
ordered `actions` list ending in `finish`. It stores neither a final canvas nor expected
rewards. Replay resets the environment with the referenced TaskSpec and executes every
action through the same interface as RL and MCP, allocating IDs normally.

| Task | Successful solution | Representative attacks |
| --- | --- | --- |
| `summer_sale` | `well_done.yaml` | `offscreen_headline.yaml`, `small_element_spam.yaml` |
| `webinar` | `well_done.yaml` | `hidden_headline.yaml`, `missing_cta.yaml` |
| `newsletter` | `well_done.yaml` | `forbidden_cta.yaml`, `tiny_logo.yaml` |
| `event` | `well_done.yaml` | `center_stacking.yaml`, `duplicate_cta.yaml` |
| `two_column` | `well_done.yaml` | `centered_layout.yaml`, `contrast_patch.yaml` |

The set deliberately includes a successful attack: duplicate CTAs where count is
unspecified still receive maximum reward. A partial background behind text ink is now
detected, reducing its reward to +0.533; soft contrast still allows a positive score.
The tiny-logo attack now fails the minimum-area presence check and receives −0.280.
These are observable limitations, not examples of successful visual design. See [REWARD_HACKING.md](REWARD_HACKING.md).

## Load and replay

```python
from marketcanvas_env.task_data import load_task, replay_trajectory, trajectory_paths

task = load_task("webinar.yaml")
for path in trajectory_paths("webinar"):
    result = replay_trajectory(path)
    print(path.stem, result["reward_breakdown"]["reward"])
```

`load_trajectory(path)` validates and returns the authored data without replaying it.
`replay_trajectory(path)` returns the task name, description, step trace, final state,
and complete reward breakdown. `trajectory_paths()` discovers all trajectories in stable
order without an index file. These APIs find bundled data independently of the working
directory, both in a checkout and an installed wheel. Documents are cached for the process
lifetime; restart after editing YAML.

```sh
uv run --locked python scripts/run_examples.py
uv run --locked python scripts/run_examples.py --task two_column
uv run --locked python scripts/run_examples.py --task webinar --trajectory missing_cta --output-dir artifacts/reward
uv run --locked python scripts/mcp_client.py --task webinar --trajectory missing_cta
```

There are two runnable utilities in `scripts/`. `run_examples.py` replays all data by
default and prints a reward table; `--task` and `--trajectory` narrow the selection.
`--output-dir` also saves PNGs and JSON reports. `mcp_client.py` replays one trajectory
through the real MCP server, defaulting to `well_done`. Both read the same YAML data;
they do not define additional example tasks or action sequences.

To add a task, author a new TaskSpec YAML with prompt metadata and supported constraints,
then add a matching trajectory directory. Pass the loaded task directly to
`env.reset(options=load_task("webinar.yaml").model_dump(mode="json"))`;
explicit constraints bypass parsing.
No reward changes or prompt-specific weights are needed. Each `constraints` field
has a TODO for a future deterministic parsing stage. The [parser extension point](REWARD.md#prompt-interpretation)
currently raises `NotImplementedError` for every prompt-only reset; saved YAML prompts
are not recognized automatically. Keep this public dataset small:
extensive malformed snapshots,
parameter sweeps, and historical comparisons belong in tests. The broader 52-case reward
regressions and historical benchmark snapshots share one
[`tests/fixtures/reward_regressions.yaml`](../tests/fixtures/reward_regressions.yaml), which
is not installed as runtime data.
Their helper code and historical reward experiments live in `tests/reward_support/`.
