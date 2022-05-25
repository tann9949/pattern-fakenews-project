from typing import List

from pythainlp.tokenize import newmm
from pythainlp.ulmfit.preprocess import (fix_html, lowercase_all, remove_space,
                                         replace_rep_nonum, replace_url,
                                         replace_wrep_post_nonum, rm_brackets,
                                         rm_useless_newlines,
                                         rm_useless_spaces, spec_add_spaces,
                                         ungroup_emoji)
from pythainlp.util import reorder_vowels


TOKENIZER_PRE_RULES = [
    fix_html,
    reorder_vowels,
    spec_add_spaces,
    rm_useless_spaces,
    rm_useless_newlines,
    rm_brackets,
    replace_url,
    replace_rep_nonum
]

TOKENIZER_POST_RULES = [
    ungroup_emoji,
    lowercase_all,
    replace_wrep_post_nonum,
    remove_space
]

def tokenize(text: str) -> List[str]:
    # run pre-processing rules before tokenize
        for rule in TOKENIZER_PRE_RULES:
            text = rule(text)

        # tokenize
        tokens = newmm.segment(text)

        # run post-processing rules after tokenize
        for rule in TOKENIZER_POST_RULES:
            tokens = rule(tokens)
        return tokens
        

