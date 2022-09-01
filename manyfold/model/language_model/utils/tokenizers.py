from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import regex as re


class BaseTokenizer(ABC):
    """
    Tokenizer abstract class.
    """

    @property
    @abstractmethod
    def vocabulary(self) -> List[str]:
        pass

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    @property
    @abstractmethod
    def standard_tokens(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def special_tokens(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def unk_token(self) -> str:
        pass

    @property
    @abstractmethod
    def pad_token(self) -> str:
        pass

    @property
    @abstractmethod
    def mask_token(self) -> str:
        pass

    @property
    @abstractmethod
    def class_token(self) -> str:
        pass

    @property
    @abstractmethod
    def eos_token(self) -> str:
        pass

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id(self.unk_token)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id(self.pad_token)

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id(self.mask_token)

    @property
    def class_token_id(self) -> int:
        return self.token_to_id(self.class_token)

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id(self.eos_token)

    @abstractmethod
    def id_to_token(self, token_id: int) -> str:
        pass

    @abstractmethod
    def token_to_id(self, token: str) -> int:
        pass

    @abstractmethod
    def tokenize(self, sequence: str) -> Tuple[List[str], List[int]]:
        """
        Tokenize a sequence and returns the list of tokens as well
        as the list of their IDs. If a single character that does not correspond
        to any token is found, it is replaced by the unk token.
        """
        pass

    @abstractmethod
    def batch_tokenize(self, sequences: List[str]) -> List[Tuple[List[str], List[int]]]:
        """
        Tokenize a batch of sequences.
        """
        pass

    def pad_tokens_batch(
        self, batch: List[Tuple[List[str], List[int]]]
    ) -> List[Tuple[List[str], List[int]]]:
        """
        Takes a batch of sequences tokens ids and returns a batch of padded sequences.

        Args:
            batch: List of tuples, each composed of a sequence's tokens and token ids.

        Returns:
            the padded list, where every sequence is padded to the maximum
            length of the batch.
        """
        lengths = [len(t[0]) for t in batch]
        maximum_length = max(lengths)
        deltas = [maximum_length - length for length in lengths]
        padded_tokens = [
            t[0] + ([self.pad_token] * delta) for t, delta in zip(batch, deltas)
        ]
        padded_tokens_ids = [
            t[1] + ([self.pad_token_id] * delta) for t, delta in zip(batch, deltas)
        ]
        return [
            (toks, toks_ids) for toks, toks_ids in zip(padded_tokens, padded_tokens_ids)
        ]


class BasicTokenizer(BaseTokenizer):
    """
    Simple tokenizer that naively extract tokens. Used for amino-acids and nucleotides.

    Args:
        standard_tokens: Standard tokens, where special tokens are omitted.
        unk_token: Unknown token.
        pad_token: Pad token.
        mask_token: Mask token.
        class_token: Class token.
        eos_token: End of speech tokens.
        prepend_cls_token: Prepend class token.
        append_eos_token: Append end of speech token.
        extra_special_tokens: (Optional) Enable the user to define optionally additional
            special tokens.
        tokens_to_ids: (Optional) Enable the user to optionally choose ids for the
            tokens. If this value is not precised, then ids are attributed automatically
            by the tokenizer during initialization.
    """

    def __init__(
        self,
        standard_tokens: List[str],
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        class_token: str = "<cls>",
        eos_token: str = "<eos>",
        prepend_cls_token: bool = False,
        append_eos_token: bool = False,
        extra_special_tokens: Optional[List[str]] = None,
        tokens_to_ids: Optional[Dict[str, int]] = None,
    ):
        special_tokens = [unk_token, pad_token, mask_token, class_token, eos_token]
        if extra_special_tokens is not None:
            special_tokens.extend(extra_special_tokens)

        self._all_tokens = special_tokens + standard_tokens
        self._standard_tokens = standard_tokens
        self._special_tokens = special_tokens

        self._unk_token = unk_token
        self._pad_token = pad_token
        self._mask_token = mask_token
        self._class_token = class_token
        self._eos_token = eos_token
        self._prepend_cls_token = prepend_cls_token
        self._append_eos_token = append_eos_token

        # Matching between tokens and ids
        if tokens_to_ids is not None:
            if set(tokens_to_ids.keys()) != set(self._all_tokens):
                raise ValueError(
                    f"Specified matching between tokens and ids, "
                    f"but some tokens are missing or mismatch. "
                    f"Got specifications for tokens: {set(tokens_to_ids.keys())} "
                    f"and expected for {set(self._all_tokens)}"
                )
            sorted_tokens = np.sort(list(tokens_to_ids.values()))
            if np.any(sorted_tokens != np.arange(len(self._all_tokens))):
                raise ValueError(
                    f"Specified matching between tokens and ids, "
                    f"but some ids are missing or mismatch. "
                    f"Got specifications for ids: {sorted_tokens} "
                    f"and expected for {np.arange(len(self._all_tokens))}"
                )
            self._tokens_to_ids = tokens_to_ids
        else:
            self._tokens_to_ids = {tok: i for i, tok in enumerate(self._all_tokens)}

        self._ids_to_tokens = {i: tok for tok, i in self._tokens_to_ids.items()}
        self._compiled_regex = re.compile("|".join(self._all_tokens + ["\S"]))  # noqa

    @property
    def vocabulary(self) -> List[str]:
        return self._all_tokens

    @property
    def standard_tokens(self) -> List[str]:
        return self._standard_tokens

    @property
    def special_tokens(self) -> List[str]:
        return self._special_tokens

    @property
    def unk_token(self) -> str:
        return self._unk_token

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def mask_token(self) -> str:
        return self._mask_token

    @property
    def class_token(self) -> str:
        return self._class_token

    @property
    def eos_token(self) -> str:
        return self._eos_token

    def id_to_token(self, token_id: int) -> str:
        try:
            token = self._ids_to_tokens[token_id]
        except KeyError:
            print("Token id not found in vocabulary")
        return token

    def token_to_id(self, token: str) -> int:
        return self._tokens_to_ids.get(token, -1)

    def tokenize(self, sequence: str) -> Tuple[List[str], List[int]]:
        """
        Tokenize a sequence and returns the list of tokens as well
        as the list of their IDs. If a single character that does not correspond
        to any token is found, it is replaced by the unk token.

        Args:
            sequence: Sequence to be tokenized.

        Returns:
            the tokenized list as well as the token ids.
        """
        tokens: List[str] = self._compiled_regex.findall(sequence)
        tokens = [
            tok if self._tokens_to_ids.get(tok) else self._unk_token for tok in tokens
        ]
        if self._prepend_cls_token:
            tokens = [self._class_token] + tokens

        if self._append_eos_token:
            tokens.append(self._eos_token)

        try:
            tokens_ids = [self._tokens_to_ids[tok] for tok in tokens]
        except KeyError:
            print(
                "Found tokens in the sequence that are not supported by the tokenizer"
            )

        return tokens, tokens_ids

    def batch_tokenize(self, sequences: List[str]) -> List[Tuple[List[str], List[int]]]:
        """
        Tokenize a batch of sequences.
        Sequences are padded to the maximum length in the batch.

        Args:
            sequences: Batch of sequences to be tokenized.

        Returns:
            batch of tokenized sequences as well as their token ids,
            where every sequence has been padded to the maximum length
            in the batch.
        """
        return self.pad_tokens_batch([self.tokenize(seq) for seq in sequences])


class NucleotidesKmersTokenizer(BasicTokenizer):
    """
    This is a tokenizer specific for nucleotide sequences,
    it considers sequence containing the tokens A, T, C, G and N.
    It always consider N as a special token, it always tokenized alone.
    """

    def __init__(
        self,
        standard_tokens: List[str],
        k_mers: int,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        class_token: str = "<cls>",
        eos_token: str = "<eos>",
        prepend_cls_token: bool = False,
        append_eos_token: bool = False,
    ):
        special_tokens = [unk_token, pad_token, mask_token]
        if prepend_cls_token:
            special_tokens.append(class_token)

        if append_eos_token:
            special_tokens.append(eos_token)

        self._all_tokens = special_tokens + standard_tokens
        self._standard_tokens = standard_tokens
        self._special_tokens = special_tokens

        self._unk_token = unk_token
        self._pad_token = pad_token
        self._mask_token = mask_token
        self._class_token = class_token
        self._eos_token = eos_token
        self._prepend_cls_token = prepend_cls_token
        self._append_eos_token = append_eos_token

        self._k_mers = k_mers

        self._tokens_to_ids = {tok: i for i, tok in enumerate(self._all_tokens)}
        self._ids_to_tokens = {i: tok for tok, i in self._tokens_to_ids.items()}

    def tokenize(self, sequence: str) -> Tuple[List[str], List[int]]:
        """
        Tokenize a sequence and returns the list of tokens as well
        as the list of their IDs. The tokenization algorithm first splits up the
        substrings of the input sequence in-between N characters.
        Then these substrings are split into pieces of length k, and if it
        is possible (edge cases) it adds up pieces of size 1.
        example :
            ATCGAATGGCGATGCAC -> ATCGA ATGGC GATGC A C

        If a single character that does not correspond
        to any token is found, an error is raised.

        Args:
            sequence: Sequence to be tokenized.

        Returns:
            the tokenized list as well as the token ids.
        """
        splitted_seq = sequence.split("N")
        len_splitted = len(splitted_seq)
        tokens, tokens_ids = [], []
        for i, split in enumerate(splitted_seq):
            chunks = [
                split[i * self._k_mers : (i + 1) * self._k_mers]
                for i in range(len(split) // self._k_mers)
            ]
            if len(split) % self._k_mers != 0:
                chunks.append(split[(len(split) // self._k_mers) * self._k_mers :])

            for chunk in chunks:
                if len(chunk) == self._k_mers:
                    tokens.append(chunk)
                else:
                    for nucl in chunk:
                        tokens.append(nucl)
            if i < len_splitted - 1:
                tokens.append("N")

        if self._prepend_cls_token:
            tokens = [self._class_token] + tokens

        if self._append_eos_token:
            tokens.append(self._eos_token)

        try:
            tokens_ids = [self._tokens_to_ids[tok] for tok in tokens]
        except KeyError:
            print(
                "Found tokens in the sequence that are not supported by"
                + f"the {self._k_mers} tokenizer]"
            )

        return tokens, tokens_ids


class FixedSizeBasicTokenizer(BasicTokenizer):
    """
    Simple tokenizer that naively extract tokens. Used for amino-acids
    and nucleotides. This tokenizer also batch tokenize batches to a
    fixed maximum length. If one of the sequence provided exceed the maximum
    length, an exception is raised.

    Args:
        standard_tokens: Standard tokens, where special tokens are omitted.
        unk_token: Unknown token.
        pad_token: Pad token.
        mask_token: Mask token.
        class_token: Class token.
        eos_token: End of speech tokens.
        prepend_cls_token: Prepend class token.
        append_eos_token: Append end of speech token.
        fixed_length: Fixed length to pad all sequences in batches
        extra_special_tokens: (Optional) Enable the user to define optionally additional
            special tokens.
        tokens_to_ids: (Optional) Enable the user to optionally choose ids for the
            tokens. If this value is not precised, then ids are attributed automatically
            by the tokenizer during initialization.
    """

    def __init__(
        self,
        standard_tokens: List[str],
        fixed_length: int,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        class_token: str = "<cls>",
        eos_token: str = "<eos>",
        prepend_cls_token: bool = False,
        append_eos_token: bool = False,
        extra_special_tokens: Optional[List[str]] = None,
        tokens_to_ids: Optional[Dict[str, int]] = None,
    ):
        BasicTokenizer.__init__(
            self,
            standard_tokens=standard_tokens,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            class_token=class_token,
            eos_token=eos_token,
            prepend_cls_token=prepend_cls_token,
            append_eos_token=append_eos_token,
            extra_special_tokens=extra_special_tokens,
            tokens_to_ids=tokens_to_ids,
        )
        self._fixed_length = fixed_length

    @property
    def fixed_length(self) -> int:
        return self._fixed_length

    def pad_tokens_batch(
        self, batch: List[Tuple[List[str], List[int]]]
    ) -> List[Tuple[List[str], List[int]]]:
        """
        Takes a batch of sequences tokens ids and returns a batch of padded sequences.

        Args:
            batch: List of tuples, each composed of a sequence's tokens and token ids.

        Returns:
            the padded list, where every sequence is padded to the maximum
            length of the batch.
        """
        lengths = [len(t[0]) for t in batch]
        maximum_length = max(lengths)
        if maximum_length > self._fixed_length:
            raise ValueError(
                "Found a sequence with length that "
                "exceeds the fixed length to tokenize."
            )
        deltas = [self._fixed_length - length for length in lengths]
        padded_tokens = [
            t[0] + ([self.pad_token] * delta) for t, delta in zip(batch, deltas)
        ]
        padded_tokens_ids = [
            t[1] + ([self.pad_token_id] * delta) for t, delta in zip(batch, deltas)
        ]
        return [
            (toks, toks_ids) for toks, toks_ids in zip(padded_tokens, padded_tokens_ids)
        ]


class FixedSizeNucleotidesKmersTokenizer(NucleotidesKmersTokenizer):
    """
    Simple tokenizer that naively extract tokens. Used for amino-acids
    and nucleotides. This tokenizer also batch tokenize batches to a
    fixed maximum length. If one of the sequence provided exceed the maximum
    length, an exception is raised.

    Args:
        standard_tokens: Standard tokens, where special tokens are omitted.
        unk_token: Unknown token.
        pad_token: Pad token.
        mask_token: Mask token.
        class_token: Class token.
        eos_token: End of speech tokens.
        prepend_cls_token: Prepend class token.
        append_eos_token: Append end of speech token.
        fixed_length: Fixed length to pad all sequences in batches
    """

    def __init__(
        self,
        standard_tokens: List[str],
        k_mers: int,
        fixed_length: int,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        class_token: str = "<cls>",
        eos_token: str = "<eos>",
        prepend_cls_token: bool = False,
        append_eos_token: bool = False,
    ):
        NucleotidesKmersTokenizer.__init__(
            self,
            standard_tokens=standard_tokens,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            class_token=class_token,
            eos_token=eos_token,
            prepend_cls_token=prepend_cls_token,
            append_eos_token=append_eos_token,
            k_mers=k_mers,
        )
        self._fixed_length = fixed_length

    @property
    def fixed_length(self) -> int:
        return self._fixed_length

    def pad_tokens_batch(
        self, batch: List[Tuple[List[str], List[int]]]
    ) -> List[Tuple[List[str], List[int]]]:
        """
        Takes a batch of sequences tokens ids and returns a batch of padded sequences.

        Args:
            batch: List of tuples, each composed of a sequence's tokens and token ids.

        Returns:
            the padded list, where every sequence is padded to the maximum
            length of the batch.
        """
        lengths = [len(t[0]) for t in batch]
        maximum_length = max(lengths)
        if maximum_length > self._fixed_length:
            raise ValueError(
                "Found a sequence with length that "
                "exceeds the fixed length to tokenize."
            )
        deltas = [self._fixed_length - length for length in lengths]
        padded_tokens = [
            t[0] + ([self.pad_token] * delta) for t, delta in zip(batch, deltas)
        ]
        padded_tokens_ids = [
            t[1] + ([self.pad_token_id] * delta) for t, delta in zip(batch, deltas)
        ]
        return [
            (toks, toks_ids) for toks, toks_ids in zip(padded_tokens, padded_tokens_ids)
        ]
