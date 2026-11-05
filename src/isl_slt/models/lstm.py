"""Complete additive-attention LSTM decoder with training teacher forcing and free beam generation."""

import torch
from torch import nn

from isl_slt.models.encoders import VisualEncoder, masked_mean


class AttentionLSTM(nn.Module):
    def __init__(self, encoder: VisualEncoder, vocabulary: int, hidden: int, embedding: int,
                 dropout: float, lexical_size: int = 0,
                 word_dropout: float = 0.0, ctc_weight: float = 0.0,
                 rank_weight: float = 0.0, rank_margin: float = 0.2):
        super().__init__()
        self.visual = encoder
        self.embedding = nn.Embedding(vocabulary, embedding, padding_idx=0)
        self.initial = nn.Linear(encoder.output_dim, hidden)
        self.key = nn.Linear(encoder.output_dim, hidden, bias=False)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.energy = nn.Linear(hidden, 1, bias=False)
        self.cell = nn.LSTMCell(embedding + encoder.output_dim, hidden)
        self.output = nn.Linear(hidden + encoder.output_dim, vocabulary)
        self.dropout = nn.Dropout(dropout)
        self.lexical_head = nn.Linear(encoder.output_dim, lexical_size) if lexical_size else None
        self.word_dropout, self.ctc_weight = word_dropout, ctc_weight
        self.ctc_head = nn.Linear(encoder.output_dim, vocabulary) if ctc_weight else None
        self.rank_weight, self.rank_margin = rank_weight, rank_margin

    def step(self, token, state, memory, keys, mask):
        hidden, cell = state
        scores = self.energy(torch.tanh(keys + self.query(hidden)[:, None])).squeeze(-1).float()
        weights = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1).to(memory.dtype)
        context = torch.bmm(weights[:, None], memory).squeeze(1)
        # Recurrent arithmetic stays FP32; projections can still use outer autocast.
        with torch.autocast(memory.device.type, enabled=False):
            hidden, cell = self.cell(torch.cat([self.dropout(self.embedding(token)), context], -1).float(),
                                     (hidden.float(), cell.float()))
        logits = self.output(torch.cat([self.dropout(hidden), context], -1))
        return logits, (hidden, cell)

    def initialize(self, memory, mask):
        hidden = torch.tanh(self.initial(masked_mean(memory, mask)))
        return hidden.float(), torch.zeros_like(hidden).float()

    def _decode(self, memory, mask, inputs, steps):
        keys, state = self.key(memory), self.initialize(memory, mask)
        logits = []
        for index in range(steps):
            output, state = self.step(inputs[:, index], state, memory, keys, mask)
            logits.append(output)
        return torch.stack(logits, 1)

    def forward(self, *, labels: torch.Tensor, **visual):
        memory, mask = self.visual(**visual)
        start = torch.full((len(memory), 1), 2, dtype=torch.long, device=memory.device)
        previous = labels[:, :-1].masked_fill(labels[:, :-1] == -100, 0)
        if self.training and self.word_dropout:
            drop = torch.rand(previous.shape, device=previous.device) < self.word_dropout
            previous = previous.masked_fill(drop & (previous > 3), 1)
        inputs = torch.cat([start, previous], 1)
        stacked = self._decode(memory, mask, inputs, labels.shape[1])
        rank = None
        if self.training and self.rank_weight and len(memory) > 1:
            # Conditional ranking: gold text must be likelier under its own video than another clip's.
            # Length-matched negative: pair each clip with its nearest-length neighbour so the
            # decoder cannot win the ranking by checking text length against video length.
            count = len(memory)
            ranked = torch.argsort(mask.sum(1))
            partner = torch.arange(count, device=memory.device) ^ 1
            partner[partner >= count] = count - 2
            order = torch.empty_like(ranked)
            order[ranked] = ranked[partner]
            negative = self._decode(memory[order], mask[order], inputs, labels.shape[1])
            valid = (labels != -100).float()
            def nll(lg):
                tok = torch.nn.functional.cross_entropy(lg.float().transpose(1, 2), labels, ignore_index=-100,
                                                        reduction="none")
                return (tok * valid).sum(1) / valid.sum(1).clamp(min=1)
            real, shuf = nll(stacked), nll(negative)
            rank = self.rank_weight * torch.relu(self.rank_margin + real - shuf).mean()
        lexical = self.lexical_head(masked_mean(memory, mask)) if self.lexical_head is not None else None
        result = {"logits": stacked, "lexical_logits": lexical}
        if rank is not None:
            result["rank_mean"] = rank
        if self.ctc_head is not None and self.training:
            log_probs = self.ctc_head(memory).float().log_softmax(-1).transpose(0, 1)
            valid = labels != -100
            targets = labels[valid & (labels > 3)]
            target_lengths = (valid & (labels > 3)).sum(1)
            ctc = torch.nn.functional.ctc_loss(log_probs, targets, mask.sum(1), target_lengths, blank=0,
                                               reduction="sum", zero_infinity=True)
            result["ctc_sum"] = self.ctc_weight * ctc
        return result

    @torch.no_grad()
    def generate(self, *, beams: int, max_new_tokens: int, length_penalty: float, **visual):
        memory, mask = self.visual(**visual)
        predictions = []
        for index in range(len(memory)):
            mem, valid = memory[index:index + 1], mask[index:index + 1]
            keys = self.key(mem)
            # Each candidate keeps its own recurrent state; completed EOS candidates remain eligible.
            candidates = [([], 0.0, self.initialize(mem, valid), False)]
            for _ in range(max_new_tokens):
                expanded = []
                for tokens, score, state, complete in candidates:
                    if complete:
                        expanded.append((tokens, score, state, True))
                        continue
                    previous = torch.tensor([tokens[-1] if tokens else 2], device=memory.device)
                    logits, next_state = self.step(previous, state, mem, keys, valid)
                    log_probs = logits.float().log_softmax(-1)[0]
                    log_probs[0], log_probs[2] = -torch.inf, -torch.inf
                    values, indices = log_probs.topk(min(beams, len(log_probs) - 2))
                    for value, token in zip(values.tolist(), indices.tolist()):
                        expanded.append((tokens + [token], score + value, next_state, token == 3))
                candidates = sorted(expanded, key=lambda c: c[1] / max(len(c[0]), 1) ** length_penalty,
                                    reverse=True)[:beams]
                if all(candidate[3] for candidate in candidates):
                    break
            predictions.append(candidates[0][0])
        output = torch.zeros(len(predictions), max(map(len, predictions)), dtype=torch.long, device=memory.device)
        for index, prediction in enumerate(predictions):
            output[index, :len(prediction)] = torch.tensor(prediction, device=memory.device)
        return output
