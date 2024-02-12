import numpy as np

from cortex.tokenization import ProteinSequenceTokenizerFast
from cortex.transforms import HuggingFaceTokenizerTransform


def test_hugging_face_tokenizer_transform():
    tokenizer = ProteinSequenceTokenizerFast()
    transform = HuggingFaceTokenizerTransform(tokenizer)
    a_id = tokenizer.vocab["A"]
    v_id = tokenizer.vocab["V"]
    c_id = tokenizer.vocab["C"]

    seq_array = np.array(
        [
            "A V A V A V C C",
            "A C V A C A",
        ]
    )

    gt_tok_id_array = [
        [a_id, v_id, a_id, v_id, a_id, v_id, c_id, c_id],
        [a_id, c_id, v_id, a_id, c_id, a_id],
    ]
    act_tok_id_array = transform(seq_array)

    assert act_tok_id_array == gt_tok_id_array
