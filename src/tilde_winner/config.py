from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    ffn_mult: float = 3.5
    dropout: float = 0.0
    rope_theta: float = 10000.0
    variant: str = "egm"

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.variant in {"baseline", "egm"}, "variant must be baseline or egm"


@dataclass(slots=True)
class DataConfig:
    n_keys: int = 64
    num_pairs: int = 32
    bos_id: int = 1
    sep_id: int = 2

    @property
    def vocab_size(self) -> int:
        return 3 + (2 * self.n_keys)

    @property
    def seq_len(self) -> int:
        # [BOS] + (k,v)*num_pairs + [SEP] + [query_k, query_v]
        return 2 * self.num_pairs + 4
