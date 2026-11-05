import sys, json, torch, torch.nn.functional as F
sys.path.insert(0, "src")
from isl_slt.config import Config
from isl_slt.workflows import evaluation_inputs
from isl_slt.data.dataset import Collator
from isl_slt.training.checkpoint import load_checkpoint
ck = sys.argv[1]
cfg = Config.model_validate(load_checkpoint(ck)["config"])
pred, ds, _ = evaluation_inputs(cfg, ck, "validation")
m, tok, dev = pred.model.eval(), pred.tokenizer, pred.device
col = Collator(tok, labels=True)
out = {"ckpt": ck}
def offdiag_cos(x):  # x: B,D
    x = F.normalize(x.float(), dim=-1); s = x @ x.T; B = s.shape[0]
    return float((s.sum() - s.diag().sum()) / (B * (B - 1)))
stats = {k: [] for k in ["ce_real", "ce_shuf", "ce_zero", "cos_visual", "cos_bridge", "cos_t5enc", "bridge_std", "bridge_rms"]}
idx = list(range(0, len(ds), max(1, len(ds) // 256)))[:256]
with torch.no_grad():
    for s in range(0, len(idx), 32):
        b = col([ds[i] for i in idx[s:s + 32]]).to(dev)
        lab = b.labels
        def ce(**v):
            o = m(labels=lab, **v)["logits"].float()
            return float(F.cross_entropy(o.reshape(-1, o.shape[-1]), lab.reshape(-1), ignore_index=-100))
        stats["ce_real"].append(ce(**b.visual))
        perm = torch.roll(torch.arange(lab.shape[0], device=dev), 1)
        shuf = {k: (v[perm] if torch.is_tensor(v) and v.shape[0] == lab.shape[0] else v) for k, v in b.visual.items()}
        stats["ce_shuf"].append(ce(**shuf))
        if hasattr(m, "t5"):
            st, emb, mask = m.encode_visual(**b.visual)
            zo = m.t5(inputs_embeds=torch.zeros_like(emb), attention_mask=mask.long(), labels=lab).logits.float()
            stats["ce_zero"].append(float(F.cross_entropy(zo.reshape(-1, zo.shape[-1]), lab.reshape(-1), ignore_index=-100)))
            enc = m.t5.encoder(inputs_embeds=emb, attention_mask=mask.long()).last_hidden_state
            mm = mask[..., None].float()
            pool = lambda x: (x * mm).sum(1) / mm.sum(1)
            stats["cos_visual"].append(offdiag_cos(pool(st))); stats["cos_bridge"].append(offdiag_cos(pool(emb)))
            stats["cos_t5enc"].append(offdiag_cos(pool(enc)))
            stats["bridge_std"].append(float(emb[mask].float().std())); stats["bridge_rms"].append(float(emb[mask].float().pow(2).mean().sqrt()))
        else:
            st, mask = m.visual(**b.visual); mm = mask[..., None].float()
            stats["cos_visual"].append(offdiag_cos((st * mm).sum(1) / mm.sum(1)))
out.update({k: round(sum(v) / len(v), 4) for k, v in stats.items() if v})
if hasattr(m, "t5"):
    E = m.t5.shared.weight.float()
    out["t5_tok_emb_std"] = round(float(E.std()), 4); out["t5_tok_emb_rms"] = round(float(E.pow(2).mean().sqrt()), 4)
    out["bridge_linear_weight_std"] = round(float(m.bridge[1].weight.std()), 5)
    out["t5_config"] = {k: getattr(m.t5.config, k) for k in ["d_model", "decoder_start_token_id", "pad_token_id", "eos_token_id", "tie_word_embeddings"]}
    out["label_example"] = [int(x) for x in lab[0][:20]]
    out["label_text"] = tok.decode([int(x) for x in lab[0] if x >= 0], skip_special_tokens=False)
print(json.dumps(out))
