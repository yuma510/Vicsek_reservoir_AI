import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vicsek_rc import new_narma_dir, save_narma_params


def generate_narma10(length, low=0.0, high=0.5, seed=555):
    rng = np.random.default_rng(seed)
    u = rng.uniform(low, high, length)
    y = np.zeros(length)

    for t in range(10, length - 1):
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * np.sum(y[t - 9:t + 1])
            + 1.5 * u[t - 9] * u[t]
            + 0.1
        )

    return u, y


def parse_args():
    parser = argparse.ArgumentParser(description="Generate NARMA10 input and target data.")
    parser.add_argument("--length", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--high", type=float, default=0.5)
    parser.add_argument("--output-dir", default="narma_data")
    parser.add_argument("--input-name",  default=None,
                        help="Output filename for input data. "
                             "Defaults to narma10_input_<low>:<high>_seed<seed>.dat")
    parser.add_argument("--target-name", default=None,
                        help="Output filename for target data. "
                             "Defaults to narma10_target_<low>:<high>_seed<seed>.dat")
    return parser.parse_args()


def main():
    args = parse_args()
    # narma_data/<YYYYMMDD_HHMMSS>/ 配下に出力（data/・reservoir_data/ と同じ規約）
    output_dir = new_narma_dir(args.output_dir)

    prefix = f"{args.low}:{args.high}_seed{args.seed}"
    input_name  = args.input_name  or f"narma10_input_{prefix}.dat"
    target_name = args.target_name or f"narma10_target_{prefix}.dat"

    u, y = generate_narma10(args.length, args.low, args.high, args.seed)
    np.savetxt(output_dir / input_name, u)
    np.savetxt(output_dir / target_name, y)
    print(f"Saved input to {output_dir / input_name}")
    print(f"Saved target to {output_dir / target_name}")

    # 使用パラメータをメタデータ JSON として出力（再現用、CLAUDE.md §5）
    params_path = save_narma_params(output_dir, input_name, target_name,
                                    args.length, args.seed, args.low, args.high)
    print(f"Saved params to {params_path}")


if __name__ == "__main__":
    main()
