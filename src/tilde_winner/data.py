import torch

from tilde_winner.config import DataConfig


def sample_batch(
    cfg: DataConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build synthetic key-value retrieval sequences.

    The model must learn to recover the value corresponding to the final query key
    from key-value pairs appearing much earlier in the context.
    """
    seq = torch.zeros((batch_size, cfg.seq_len), dtype=torch.long, device=device)
    key_offset = 3
    value_offset = 3 + cfg.n_keys

    for b in range(batch_size):
        key_ids = torch.randperm(cfg.n_keys, device=device)[: cfg.num_pairs]
        value_ids = torch.randperm(cfg.n_keys, device=device)[: cfg.num_pairs]
        key_tokens = key_ids + key_offset
        value_tokens = value_ids + value_offset

        seq[b, 0] = cfg.bos_id
        start = 1
        for i in range(cfg.num_pairs):
            seq[b, start + (2 * i)] = key_tokens[i]
            seq[b, start + (2 * i) + 1] = value_tokens[i]

        query_pair_idx = torch.randint(0, cfg.num_pairs, (1,), device=device).item()
        query_key = key_tokens[query_pair_idx]
        query_value = value_tokens[query_pair_idx]

        sep_idx = 1 + (2 * cfg.num_pairs)
        seq[b, sep_idx] = cfg.sep_id
        seq[b, sep_idx + 1] = query_key
        seq[b, sep_idx + 2] = query_value

    x = seq[:, :-1]
    y = seq[:, 1:]
    answer_pos = x.size(1) - 1
    return x, y, answer_pos
