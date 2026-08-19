"""exp86b — PPV 위임 집계가 회수 구간에서 실제로 구제를 만드는가. (GPU 임베딩 ~5분)

G1(기하 전제)은 통과했다: gold 소수파가 더 tight (66.7%, CI 0 제외, 길이 통제 유지).
결정적 관찰 — 지배오답 내부 유사도(0.9734) ≈ 교차 유사도(0.9736): **지배오답은 클리크가
아니다.** gold만 약한 클리크(0.9788 vs 교차 0.9736)다. dominant-set 동역학이 좋아하는 구조.

여기서는 43문항(회수 구간 텍스트 덤프)에서 실제 집계를 돌린다:
  · majority / 현행 가중 (새 덤프 기준선)
  · 응집도 가중: score(a) = count^0.5 · within(a)^γ  (γ ∈ {1, 10, 50} — 유사도 대역이
    0.97 부근이라 지수를 키워야 차이가 증폭된다; 사전 고정 3값, 사후 탐색 아님)
  · replicator dynamics dominant set: x ← x∘(Sx)/(xᵀSx), 정상 분포 질량으로 답 선택
⚠️ 이 덤프는 회수 구간뿐이라 **rescue만 측정 가능하고 harm은 측정 불가** — 결과 해석에 명시.

사용: uv run python remote/ppv_delegation_test.py
"""
import json
import math
from collections import Counter, defaultdict

SRC = 'results/ppv_texts.jsonl'
GAMMAS = [1, 10, 50]


def main():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in open(SRC, encoding='utf-8')]
    print('exp86b — PPV 위임 집계 · %d문항 (회수 구간 — harm 측정 불가)' % len(rows))

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-3B-Instruct', torch_dtype=torch.bfloat16,
        device_map='cuda', output_hidden_states=True)
    model.eval()

    @torch.no_grad()
    def embed(texts):
        out = []
        for i in range(0, len(texts), 8):
            enc = tok(texts[i:i + 8], return_tensors='pt', padding=True,
                      truncation=True, max_length=1024).to('cuda')
            h = model(**enc).hidden_states[-1].float()
            m = enc.attention_mask.unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(out, 0)

    def wvote(ss):
        v = [(s['ans'], s['logp']) for s in ss if s.get('ans') is not None]
        if not v:
            return None
        m = max(lp for _, lp in v)
        w = defaultdict(float)
        for a, lp in v:
            w[a] += math.exp((lp - m) * 2.0)
        return max(w, key=w.get)

    stats = Counter()
    per_method_hits = Counter()
    details = []
    emb_store = {}
    for r in rows:
        ss = [s for s in r['samples'] if s.get('ans') is not None]
        if len(ss) < 4:
            continue
        g = r['gold']
        V = embed([s['text'] for s in ss])
        emb_store[r['id']] = V.astype('float16')
        ans = [s['ans'] for s in ss]
        S = np.clip(V @ V.T, 0, None)
        np.fill_diagonal(S, 0.0)

        res = {}
        res['majority'] = Counter(ans).most_common(1)[0][0]
        res['weighted(현행)'] = wvote(ss)
        # 응집도 가중
        groups = defaultdict(list)
        for i, a in enumerate(ans):
            groups[a].append(i)
        for gam in GAMMAS:
            sc = {}
            for a, idx in groups.items():
                if len(idx) >= 2:
                    sub = S[np.ix_(idx, idx)]
                    w = sub.sum() / (len(idx) * (len(idx) - 1))
                else:
                    w = 0.9      # 단독 표본은 응집도 정의 불가 — 낮은 기본값 (사전 고정)
                sc[a] = (len(idx) ** 0.5) * (w ** gam)
            res['coh γ=%d' % gam] = max(sc, key=sc.get)
        # replicator dynamics
        x = np.full(len(ss), 1.0 / len(ss))
        for _ in range(300):
            y = x * (S @ x)
            t = y.sum()
            if t <= 0:
                break
            y /= t
            if np.abs(y - x).sum() < 1e-10:
                x = y
                break
            x = y
        mass = defaultdict(float)
        for i, a in enumerate(ans):
            mass[a] += x[i]
        res['replicator'] = max(mass, key=mass.get)

        stats['n'] += 1
        for m2, a in res.items():
            per_method_hits[m2] += (a == g)
        details.append((r['id'], g, res, Counter(ans)))

    np.savez_compressed('results/ppv_emb.npz',
                        **{k: v for k, v in emb_store.items()})
    n = stats['n']
    print('유효 %d문항 (구 덤프 기준 회수 구간; 새 덤프에서 기준선이 이미 맞히는 문항 포함)' % n)
    print()
    print('%-18s %8s' % ('방법', '정답 수'))
    base = per_method_hits['weighted(현행)']
    for m2 in ['majority', 'weighted(현행)'] + ['coh γ=%d' % g for g in GAMMAS] + ['replicator']:
        print('%-18s %5d/%d   (현행 대비 %+d)' % (m2, per_method_hits[m2], n,
                                              per_method_hits[m2] - base))
    print()
    # 현행이 틀리는 문항에서의 구제/현행이 맞히는 문항에서의 훼손 (이 덤프 내에서)
    for m2 in ['coh γ=%d' % g for g in GAMMAS] + ['replicator']:
        resc = harm = 0
        for _, g, res, _ in details:
            b = res['weighted(현행)'] == g
            o = res[m2] == g
            resc += (not b) and o
            harm += b and (not o)
        print('%-18s 구제 %2d · 훼손 %2d (이 덤프 안에서만)' % (m2, resc, harm))
    print()
    print('임베딩 저장: results/ppv_emb.npz (후속 오프라인 분석용)')
    print('⚠️ harm의 진짜 측정은 base정답 문항 텍스트가 필요하다 — 이 덤프에는 없다.')


if __name__ == '__main__':
    main()
