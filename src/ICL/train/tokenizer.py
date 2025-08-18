"""Custom tokenizer for RHM (Random Hierarchy Model) training and evaluation."""

from pathlib import Path

from transformers import PreTrainedTokenizer


class RHMTokenizer(PreTrainedTokenizer):
    """Custom tokenizer for RHM with direct integer-to-token mapping."""

    def __init__(
        self,
        vocab_size: int = 37,
        pad_token: str = "<pad>",
        eos_token: str = "<eos>",
        sep_token: str = "<sep>",
        mask_token: str = "<mask>",
        unk_token: str = "<unk>",
        **kwargs,
    ):
        """Initialize RHM tokenizer with controlled vocabulary.

        Args:
            vocab_size: Size of RHM vocabulary (default 37)
            pad_token: Padding token string
            eos_token: End of sequence token string
            sep_token: Separator token string
            mask_token: Mask token for MLM
            unk_token: Unknown token string

        """
        # Store vocab size in private attribute to avoid property conflict
        self._rhm_vocab_size = vocab_size

        # Create vocabulary: integers + special tokens
        self._vocab = {}
        self._ids_to_tokens = {}

        # Add integer tokens (0 to vocab_size-1)
        for i in range(vocab_size):
            token = str(i)
            self._vocab[token] = i
            self._ids_to_tokens[i] = token

        # Add special tokens
        special_tokens = {
            pad_token: vocab_size,
            eos_token: vocab_size + 1,
            sep_token: vocab_size + 2,
            mask_token: vocab_size + 3,
            unk_token: vocab_size + 4,
        }

        for token, token_id in special_tokens.items():
            self._vocab[token] = token_id
            self._ids_to_tokens[token_id] = token

        # Initialize parent class
        super().__init__(
            pad_token=pad_token,
            eos_token=eos_token,
            sep_token=sep_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab(self) -> dict[str, int]:
        """Return vocabulary mapping."""
        return self._vocab

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size including special tokens."""
        return len(self._vocab)

    @property
    def rhm_vocab_size(self) -> int:
        """Return original RHM vocabulary size (without special tokens)."""
        return self._rhm_vocab_size

    def get_vocab(self) -> dict[str, int]:
        """Get vocabulary for compatibility."""
        return self._vocab

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into tokens."""
        # Split by whitespace and convert each part
        tokens = []
        for part in text.strip().split():
            if part in self._vocab:
                tokens.append(part)
            else:
                # Try to parse as integer
                try:
                    int_val = int(part)
                    if 0 <= int_val < self._rhm_vocab_size:
                        tokens.append(str(int_val))
                    else:
                        tokens.append(self.unk_token)
                except ValueError:
                    tokens.append(self.unk_token)
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        return self._vocab.get(token, self._vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token."""
        return self._ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Convert tokens back to string."""
        return " ".join(tokens)

    def encode_sequence(self, sequence: list[int]) -> list[int]:
        """Encode RHM integer sequence directly to token IDs."""
        token_ids = []
        for val in sequence:
            if 0 <= val < self._rhm_vocab_size:
                token_ids.append(val)  # Direct mapping for integers
            else:
                token_ids.append(self._vocab[self.unk_token])
        return token_ids

    def decode_sequence(self, token_ids: list[int]) -> list[int]:
        """Decode token IDs back to RHM integer sequence."""
        sequence = []
        for token_id in token_ids:
            if 0 <= token_id < self._rhm_vocab_size:
                sequence.append(token_id)  # Direct mapping for integers
            # Skip special tokens in decoded sequence
        return sequence

    def add_special_tokens_to_sequence(
        self, sequence: list[int], add_eos: bool = True, add_sep: bool = False
    ) -> list[int]:
        """Add special tokens to RHM sequence."""
        result = sequence.copy()

        if add_sep:
            result.append(self.sep_token_id)

        if add_eos:
            result.append(self.eos_token_id)

        return result

    def format_icl_prompt(self, examples: list[tuple[list[int], list[int]]], query: list[int]) -> list[int]:
        """Format in-context learning prompt using RHM sequences."""
        prompt = []

        # Add examples
        for input_seq, output_seq in examples:
            prompt.extend(input_seq)
            prompt.append(self.sep_token_id)  # Separator between input and output
            prompt.extend(output_seq)
            prompt.append(self.eos_token_id)  # End of example

        # Add query
        prompt.extend(query)
        prompt.append(self.sep_token_id)  # Indicate start of expected output

        return prompt

    def save_vocabulary(self, save_directory: str | Path, filename_prefix: str | None = None) -> tuple[str]:
        """Save vocabulary to file."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        if filename_prefix is None:
            filename_prefix = "vocab"

        vocab_file = save_directory / f"{filename_prefix}.json"

        import json

        with vocab_file.open("w", encoding="utf-8") as f:
            json.dump(self._vocab, f, indent=2)

        return (str(vocab_file),)
