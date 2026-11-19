"""Use the trained genre-aware model to list unseen movies for one user."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from neurorec.data import load_data
from neurorec.training import make_model, raw_predict


def main():
    """Load the trained genre-aware model and show recommendations for a user seen in training."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Recommend movies to a given MovieLens user')
    parser.add_argument('--user-id', type=int, required=True, help='Raw MovieLens user ID')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--results', default='results/full')
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be greater than 0')
    os.chdir(Path(__file__).resolve().parent)
    # Rebuild the network from the config saved at training time so structure matches weights.
    config = json.loads((Path(args.results) / 'config.json').read_text(encoding='utf-8'))
    checkpoint = Path(args.results) / 'checkpoints' / f'ContentAware_seed{config["seeds"][0]}.pt'
    if not checkpoint.exists():
        parser.error('No model checkpoint found; please run main.py first')
    torch.set_num_threads(config['threads'])
    data = load_data(config['data_dir'])
    # CLI takes the raw user ID; the network uses 0-based contiguous indices internally.
    matches = np.flatnonzero(data.user_ids == args.user_id)
    if not len(matches) or data.train_user_counts[matches[0]] == 0:
        parser.error('This user has no training ratings; please choose a user seen in training '
                     '(e.g. pick one from the data statistics).')
    user = int(matches[0])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = make_model('ContentAware', data, config)
    model.load_state_dict(saved['state_dict'])
    # Demo mode excludes all movies the user rated anywhere; not used for test metrics.
    observed = set()
    for part in [data.train, data.val, data.test]:
        observed.update(part.loc[part.user == user, 'item'].tolist())
    items = np.array([item for item in np.flatnonzero(data.train_item_counts > 0)
                      if int(item) not in observed], dtype=np.int64)
    scores = raw_predict(model, np.full(len(items), user, dtype=np.int64), items)
    # Sort by raw score first, then by movie index to break ties; clip to 1–5 only for display.
    top = np.lexsort((items, -scores))[:args.top_k]
    movies = data.movies.set_index('item')
    print(f'Movie recommendations for user {args.user_id} (scores are model estimates):')
    for rank, index in enumerate(top, 1):
        movie = movies.loc[int(items[index])]
        print(f'{rank:2d}. {movie["title"]} | {movie["genres"]} | '
              f'predicted rating {np.clip(scores[index], 1, 5):.2f}')


if __name__ == '__main__':
    main()