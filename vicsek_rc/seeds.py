"""試行番号 b からシミュレーションの seed を決める（2026-10-07）。

規則は `configs/seed_policy.json` にある。

- ``random64``（既定）: numpy の SeedSequence で、親の乱数の種（master_entropy）と試行番号 b から
  63 ビットの seed を 3 つ（seed_noise, seed_pos, seed_nf）作る。b が同じなら毎回同じ値になり、
  試行どうし・用途どうしで値が重ならない。
- ``legacy``: 2026-10-06 以前の規則 ``seed_X = b + legacy_offsets[X]``（b+2, b+3, b+6）。
  既存データを評価し直すときに使う。この規則では、試行 b の seed_pos と試行 b+1 の seed_noise が
  同じ値になる（同じ乱数列を別の試行・別の用途で使ってしまう）。

NARMA 入力の seed（= b）は別の乱数生成器（numpy）なので、この規則の対象外。
"""
import json
from pathlib import Path

import numpy as np

ROLES = ("seed_noise", "seed_pos", "seed_nf")
_POLICY_PATH = Path(__file__).resolve().parent.parent / "configs" / "seed_policy.json"
_MAX_SEED = (1 << 63) - 1          # C 側は long（%ld）で読むので 63 ビットに収める
_scheme_override = None


def load_seed_policy(path=None) -> dict:
    return json.loads(Path(path or _POLICY_PATH).read_text())


def set_seed_scheme(scheme) -> None:
    """スクリプトの CLI（--seed-scheme）から規則を切り替える。None なら seed_policy.json の既定に戻す。"""
    global _scheme_override
    if scheme not in (None, "random64", "legacy"):
        raise ValueError(f"unknown seed scheme: {scheme}")
    _scheme_override = scheme


def current_scheme(policy=None) -> str:
    return _scheme_override or (policy or load_seed_policy())["scheme"]


def trial_seeds(b: int, scheme=None, policy=None) -> dict:
    """試行番号 b の {seed_noise, seed_pos, seed_nf} を返す。"""
    policy = policy or load_seed_policy()
    scheme = scheme or current_scheme(policy)
    if scheme == "legacy":
        return {r: int(b) + int(policy["legacy_offsets"][r]) for r in ROLES}
    if scheme != "random64":
        raise ValueError(f"unknown seed scheme: {scheme}")
    ss = np.random.SeedSequence(entropy=int(policy["master_entropy"]), spawn_key=(int(b),))
    vals = ss.generate_state(len(ROLES), dtype=np.uint64) & np.uint64(_MAX_SEED)
    vals = [int(v) if int(v) > 0 else 1 for v in vals]
    return dict(zip(ROLES, vals))


def check_unique(trials, scheme=None) -> None:
    """指定した試行番号の seed が、試行・用途をまたいですべて違うことを確かめる。"""
    seen = {}
    for b in trials:
        for r, v in trial_seeds(b, scheme).items():
            if v in seen:
                raise ValueError(f"seed collision: b={b} {r}={v} also used by {seen[v]}")
            seen[v] = (b, r)
