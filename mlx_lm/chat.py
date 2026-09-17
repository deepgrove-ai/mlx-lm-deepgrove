# Copyright © 2023-2024 Apple Inc.

import argparse

import mlx.core as mx

from .cli_ui import ChatUI
from .generate import stream_generate
from .models.cache import make_prompt_cache
from .sample_utils import make_sampler
from .utils import load, sharded_load

DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_XTC_PROBABILITY = 0.0
DEFAULT_XTC_THRESHOLD = 0.0
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


class _ChatSession:
    """Reuse KV state only for an exact prefix of the rendered conversation."""

    def __init__(self, model, tokenizer, max_kv_size=None, system_prompt=None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_kv_size = max_kv_size
        self.system_prompt = system_prompt
        self.reset()

    def reset(self):
        self.messages = (
            [{"role": "system", "content": self.system_prompt}]
            if self.system_prompt is not None
            else []
        )
        self.cache_tokens = []
        self.prompt_cache = make_prompt_cache(self.model, self.max_kv_size)

    def prepare(self, query):
        self.messages = self.messages + [{"role": "user", "content": query}]
        # Preserve content prefilled by the template, such as Maple's <think>.
        empty_assistant = self.tokenizer.apply_chat_template(
            self.messages + [{"role": "assistant", "content": ""}],
            add_generation_prompt=False,
        )
        full_prompt = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
        )
        prefix = 0
        for a, b in zip(empty_assistant, full_prompt):
            if a != b:
                break
            prefix += 1
        self.assistant_prefix = full_prompt[prefix:]

        if full_prompt[: len(self.cache_tokens)] != self.cache_tokens:
            self.prompt_cache = make_prompt_cache(self.model, self.max_kv_size)
            self.cache_tokens = []
        prompt = full_prompt[len(self.cache_tokens) :]
        self.cache_tokens = list(full_prompt)
        return prompt

    def finish(self, tokens):
        # generate_step's lookahead has evaluated every returned token,
        # including EOS. The next template supplies any separator after EOS.
        self.cache_tokens.extend(tokens)
        content_tokens = (
            tokens[:-1]
            if tokens and tokens[-1] in self.tokenizer.eos_token_ids
            else tokens
        )
        content = self.tokenizer.decode(self.assistant_prefix + content_tokens)
        self.messages = self.messages + [{"role": "assistant", "content": content}]


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--xtc-probability",
        type=float,
        default=DEFAULT_XTC_PROBABILITY,
        help="Probability of XTC sampling to happen each next token",
    )
    parser.add_argument(
        "--xtc-threshold",
        type=float,
        default=0.0,
        help="Thresold the probs of each next token candidate to be sampled by XTC",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="PRNG seed",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    parser.add_argument(
        "--flash-head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Maple only: use the approximate FlashHead output layer. Faster "
        "decode, approximate token stream. Omit to follow the checkpoint config.",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    group = mx.distributed.init()
    rank = group.rank()
    pipeline_group = group if args.pipeline else None
    tensor_group = group if not args.pipeline else None

    mx.random.seed(args.seed)

    if group.size() > 1:
        if args.adapter_path:
            parser.error("Adapters not supported in distributed mode")
        model, tokenizer = sharded_load(
            args.model,
            pipeline_group,
            tensor_group,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        model_config = {}
        if args.flash_head is not None:
            model_config["use_flash_head"] = args.flash_head
        model, tokenizer = load(
            args.model,
            adapter_path=args.adapter_path,
            tokenizer_config={"trust_remote_code": args.trust_remote_code},
            model_config=model_config,
            trust_remote_code=args.trust_remote_code,
        )

    with ChatUI(args, rank=rank) as ui:
        session = _ChatSession(model, tokenizer, args.max_kv_size, args.system_prompt)
        while True:
            query = ui.prompt()
            if query == "q":
                ui.say_bye()
                break
            if query == "r":
                session.reset()
                ui.say_reset()
                continue
            if query == "h":
                ui.say_help()
                continue
            prompt = session.prepare(query)
            last_response = None
            tokens = []
            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                sampler=make_sampler(
                    args.temp,
                    args.top_p,
                    xtc_threshold=args.xtc_threshold,
                    xtc_probability=args.xtc_probability,
                    xtc_special_tokens=(
                        tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
                    ),
                ),
                prompt_cache=session.prompt_cache,
            ):
                ui.stream_token(response.text)
                tokens.append(response.token)
                last_response = response
            session.finish(tokens)
            ui.end_turn(last_response)


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.chat...` directly is deprecated."
        " Use `mlx_lm.chat...` or `python -m mlx_lm chat ...` instead."
    )
    main()
