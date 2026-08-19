"""exp86 2단계 — PPV 전제 검증: 정답 소수파가 추론 공간에서 더 tight한가. (GPU 약 5~10분)

임베딩 = Qwen2.5-3B 자신의 마지막 은닉 상태를 attention mask 가중 평균 풀링 (새 의존성 없음).
게이트 G1(사전등록): within(gold) > within(지배오답)인 문항 ≥60% AND 평균 차이 부트스트랩
95% CI가 0 제외 AND 길이 통제 후 부호 유지.

사용: uv run python remote/analyze_ppv_geometry.py
"""
import json
import math
import random
from collections import Counter

random.seed(0)
SRC = 'results/ppv_texts.jsonl'


def main():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in open(SRC, encoding='utf-8')]
    print('exp86 — PPV 기하 진단 · %d문항' % len(rows))

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
            h = model(**enc).hidden_states[-1].float()          # [B, T, D]
            m = enc.attention_mask.unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            v = torch.nn.functional.normalize(v, dim=-1)
            out.append(v.cpu().numpy())
        return np.concatenate(out, 0)

    def within(vs):
        if len(vs) < 2:
            return None
        s, n = 0.0, 0
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                s += float(vs[i] @ vs[j])
                n += 1
        return s / n

    per = []          # (id, wg, wd, cross, len_g, len_d, |G|, |D|)
    noise = []        # 지배오답 반분 노이즈 바닥
    for r in rows:
        g = r['gold']
        ss = [s for s in r['samples'] if s.get('ans') is not None]
        c = Counter(s['ans'] for s in ss)
        wrong = [(a, n) for a, n in c.items() if a != g]
        if c.get(g, 0) < 2 or not wrong:
            continue
        dom = max(wrong, key=lambda x: x[1])[0]
        if c[dom] < 2:
            continue
        G = [s for s in ss if s['ans'] == g]
        D = [s for s in ss if s['ans'] == dom]
        vg = embed([s['text'] for s in G])
        vd = embed([s['text'] for s in D])
        wg, wd = within(vg), within(vd)
        cross = float(np.mean(vg @ vd.T))
        per.append((r['id'], wg, wd, cross,
                    sum(s['len'] for s in G) / len(G),
                    sum(s['len'] for s in D) / len(D), len(G), len(D)))
        # 노이즈 바닥: 지배오답 ≥4표면 반분 유사도 차이
        if len(D) >= 4:
            for _ in range(5):
                idx = list(range(len(D)))
                random.shuffle(idx)
                h1 = vd[idx[:len(D) // 2]]
                h2 = vd[idx[len(D) // 2:]]
                w1, w2 = within(h1), within(h2)
                if w1 is not None and w2 is not None:
                    noise.append(abs(w1 - w2))

    print('양쪽 ≥2표 충족: %d문항 (평균 gold %.1f표 / 지배오답 %.1f표)'
          % (len(per), sum(p[6] for p in per) / len(per), sum(p[7] for p in per) / len(per)))
    print()
    wg = [p[1] for p in per]
    wd = [p[2] for p in per]
    cr = [p[3] for p in per]
    dif = [a - b for a, b in zip(wg, wd)]
    win = sum(1 for d in dif if d > 0)
    print('답내 평균 코사인 유사도:')
    print('  gold 소수파   %.4f' % (sum(wg) / len(wg)))
    print('  지배오답      %.4f' % (sum(wd) / len(wd)))
    print('  교차(G-D)     %.4f' % (sum(cr) / len(cr)))
    print('  gold가 더 tight한 문항: %d/%d = %.1f%%' % (win, len(per), 100 * win / len(per)))
    bs = []
    for _ in range(2000):
        s = [dif[random.randrange(len(dif))] for _ in range(len(dif))]
        bs.append(sum(s) / len(s))
    bs.sort()
    print('  평균 차이 %.4f · 부트스트랩 95%% CI [%.4f, %.4f]'
          % (sum(dif) / len(dif), bs[50], bs[1949]))
    if noise:
        print('  노이즈 바닥(지배오답 반분 |차이|): 평균 %.4f' % (sum(noise) / len(noise)))
    print()

    # 길이 교란
    ldif = [p[4] - p[5] for p in per]
    ml = sum(ldif) / len(ldif)
    num = sum((d - sum(dif) / len(dif)) * (l - ml) for d, l in zip(dif, ldif))
    den = math.sqrt(sum((d - sum(dif) / len(dif)) ** 2 for d in dif)
                    * sum((l - ml) ** 2 for l in ldif))
    print('길이 교란 확인: gold 평균 %d토큰 vs 지배오답 %d토큰'
          % (sum(p[4] for p in per) / len(per), sum(p[5] for p in per) / len(per)))
    print('  유사도차-길이차 상관 r = %.3f' % (num / den if den else 0.0))
    sub = [p for p in per if 200 <= p[4] <= 700 and 200 <= p[5] <= 700]
    if len(sub) >= 8:
        sd = [p[1] - p[2] for p in sub]
        sw = sum(1 for d in sd if d > 0)
        print('  길이 대역 [200,700] 제한 (%d문항): gold 더 tight %d/%d = %.1f%%, 평균 차이 %.4f'
              % (len(sub), sw, len(sub), 100 * sw / len(sub), sum(sd) / len(sd)))
    print()
    ci_lo, ci_hi = bs[50], bs[1949]
    g1 = (win / len(per) >= 0.60) and (ci_lo > 0 or ci_hi < 0) and (sum(dif) / len(dif) > 0)
    print('사전등록 G1 판정: 승률 %.1f%% (기준 ≥60%%) · CI 0 제외 %s · 방향 %s'
          % (100 * win / len(per), 'O' if (ci_lo > 0 or ci_hi < 0) else 'X',
             '+' if sum(dif) / len(dif) > 0 else '−'))
    print('→ G1 %s' % ('통과 — PPV 위임 설계로 확장 논의' if g1 else '실패 — semantic/기하 축 종료'))


if __name__ == '__main__':
    main()
