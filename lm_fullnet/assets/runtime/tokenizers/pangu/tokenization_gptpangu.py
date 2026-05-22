import logging
from shutil import copyfile
import os

from transformers.tokenization_utils import PreTrainedTokenizer

import torch
import sentencepiece
import jieba
from typing import Optional

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.model"}


class GPTPanguTokenizer(PreTrainedTokenizer):
    # Ref: https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha/src/branch/master/tokenization_jieba.py
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
            self,
            vocab_file,
            eos_token="<eot>",
            **kwargs
    ):

        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.Load(str(vocab_file))
        self.vocab_file = vocab_file
        self.translator = str.maketrans(" \n", "\u2582\u2583")

        # special token ids
        self.eos_token_id = self.sp.piece_to_id("<eot>")

        super().__init__(
            eos_token=eos_token,
            **kwargs,
        )

    def tokenize(self, text, **kwargs):
        """ Tokenize a string. """
        seg_list = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        new_seg = " ".join(seg_list)
        return self.sp.encode(new_seg)

    def convert_tokens_to_ids(self, tokens):
        return tokens

    def convert_ids_to_tokens(self, ids):
        return self.decode(ids)

    def decode(self, tokens, **kwargs):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        text = self.sp.decode(tokens)
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text

    @property
    def vocab_size(self):
        return len(self.sp)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)
