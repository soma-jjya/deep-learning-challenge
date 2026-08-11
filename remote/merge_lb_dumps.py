"""여러 리더보드 덤프 파일을 문항별로 병합 — 분산 축소 스택 실험(exp41)용.

서로 다른 시드로 뽑은 n=32 표본 3세트를 합쳐 96표짜리 병합 덤프를 만든다.
GPU 불필요, 수 초.

사용: uv run python remote/merge_lb_dumps.py --out results/lb_samples_s3x32.jsonl \
    --sources results/lb_samples.jsonl:32 results/lb_samples_seed43.jsonl:32 results/lb_samples_seed44.jsonl:32
각 소스는 path[:n] 형식 (n개 표본만 사용, 생략 시 전부)
"""
import argparse
import json
from collections import defaultdict


def load(path, n=None):
    rows = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        rows[r['id']] = r['samples'][:n] if n else r['samples']
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', required=True, help='path:n 형식 (n 생략 시 전부 사용)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    merged = defaultdict(list)
    for spec in args.sources:
        if ':' in spec:
            path, n = spec.rsplit(':', 1)
            n = int(n)
        else:
            path, n = spec, None
        rows = load(path, n)
        print(f'{path}: {len(rows)}문항, 문항당 {n or "전부"}표 사용')
        for pid, samples in rows.items():
            merged[pid].extend(samples)

    with open(args.out, 'w', encoding='utf-8') as f:
        for pid, samples in merged.items():
            f.write(json.dumps({'id': pid, 'samples': samples}, ensure_ascii=False) + chr(10))

    counts = set(len(v) for v in merged.values())
    print(f'저장: {args.out} ({len(merged)}문항, 문항당 표 수={counts})')


if __name__ == '__main__':
    main()
