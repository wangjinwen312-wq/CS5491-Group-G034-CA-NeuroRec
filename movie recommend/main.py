"""Run the assignment: python main.py; first check the pipeline with python main.py --quick."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    """Read CLI arguments and config, then choose data prep, full experiment, or re-plotting."""
    # Force UTF-8 on Windows terminals to avoid garbled Chinese movie titles and messages.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='MovieLens 1M movie recommendation course assignment')
    parser.add_argument('--quick', action='store_true',
                        help='Debug: 1 seed, 2 epochs; does not replace the formal experiment')
    parser.add_argument('--config', help='Path to a custom JSON config file')
    parser.add_argument('--prepare', action='store_true', help='Only download and check the data')
    parser.add_argument('--plot-only', action='store_true', help='Regenerate plots from existing results')
    args = parser.parse_args()
    # Anchor to the script directory so config and data resolve when launched elsewhere.
    os.chdir(Path(__file__).resolve().parent)
    # --quick only shortens the debug run; formal params and output dir come from the full config.
    config_path = args.config or ('configs/quick.json' if args.quick else 'configs/full.json')
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    # All three modes share one config, so data prep or re-plotting needs no retraining.
    if args.prepare:
        from neurorec.data import load_data, summarize
        print(json.dumps(summarize(load_data(config['data_dir'])), ensure_ascii=False, indent=2))
    elif args.plot_only:
        from neurorec.plots import plot_all
        plot_all(Path(config['results_dir']))
    else:
        from neurorec.experiment import run_experiment
        run_experiment(config)


if __name__ == '__main__':
    main()