from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalArguments:
    """
    Configuration for running the evaluation.
    """
    use_self_infill: Optional[bool] = field(
        default=False,
        metadata={"help": "Enabling self-infilling generation"},
    )
    self_infill_tau: Optional[float] = field(
        default=0.25, 
        metadata={"help": "Self-infilling threshold for interruption."}
    )
    self_infill_max_iters: Optional[int] = field(
        default=1, 
        metadata={"help": "Max number of iterations for self-infilling looping."}
    )
    self_infill_suffix_split_mode: Optional[str] = field(
        default="vanilla", 
        metadata={"help": "Suffix splitting strategy for self-infilling looping."}
    )
    self_infill_random_interruption: Optional[bool] = field(
        default=False,
        metadata={"help": "stochastically thresholding the probability for self-infilling interruption"},
    )
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Sample from the language model's output distribution."},
    )
    temperature: Optional[float] = field(
        default=0.2, metadata={"help": "Sampling temperature used for generation."}
    )
    top_k: Optional[int] = field(
        default=0, metadata={"help": "Top-k parameter used for generation."}
    )
    top_p: Optional[float] = field(
        default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."}
    )
    n_samples: Optional[int] = field(
        default=1,
        metadata={"help": "Number of completions to generate for each sample."},
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed used for evaluation."}
    )
