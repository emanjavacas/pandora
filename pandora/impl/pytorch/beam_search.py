
import torch

from pandora.impl.pytorch.utils import multiple_index_select


def beam_search(score_fn, output_fn, inp_state, end_sym, bwidth, max_seq_len):
    """
    :param score_fn: (inp, prev_output) -> scores, output; where:
        inp: (batch (* bwidth))
        prev_output: any extra output returned by score_fn in the previous step
        scores: (batch x (bwidth or 1) x vocab)
        output: any extra output to be use in the next decoding step
    :param output_fn: (prev_output, source_beam) -> prev_output, function
        that possibly rearranges model state according to the selected beams
    :param inp_state: (batch) input for the first step
    :param end_sym: integer encoding the end of sequence symbol
    :param bwidth: number of entries in the beam
    :param max_seq_len: maximum length of the decoded sequence

    :return decoded: (batch x beam x seq) LongTensor with the decoded sequences
    :return topk_scores: (batch x beam) FloatTensor with the decoded scores
    """
    batch = inp_state.size(0)
    # compute initial scores (batch x 1 x vocab)
    scores, prev_output = score_fn(inp_state, None)
    vocab = scores.size(2)
    # (pool beams in same batch: batch x bwidth)
    topk_scores, topk_syms = scores.view(batch, -1).topk(k=bwidth)

    # create (float) mask (batch x bwidth)
    mask = (topk_syms != end_sym).float()

    # create output accumulator
    decoded = [topk_syms]

    for _ in range(max_seq_len - 1):
        # check mask and return if all done
        if mask.sum() == 0:
            break

        # get new scores
        scores, prev_output = score_fn(topk_syms.view(-1), prev_output)
        # (zero scores for ended beams)
        scores = scores * mask.unsqueeze(2)
        # (accumulate scores from previous step)
        scores = scores + topk_scores.unsqueeze(2)
        # (pool beams in same batch and extract best k)
        topk_scores, topk_syms = scores.view(batch, -1).topk(k=bwidth)
        # (pick source beam and best hypotheses per batch)
        source_beam, topk_syms = topk_syms / vocab, topk_syms % vocab

        # repackage by source beam
        # mask = multiple_index_select(mask, source_beam)
        # topk_scores = multiple_index_select(topk_scores, source_beam)
        prev_output = output_fn(prev_output, source_beam)

        # update mask
        mask = mask * (topk_syms != end_sym).float()

        decoded.append(topk_syms)

    # (seq x batch x beam) -> (batch x beam x seq)
    decoded = torch.stack(decoded).transpose(0, 1).transpose(1, 2)

    return decoded, topk_scores
