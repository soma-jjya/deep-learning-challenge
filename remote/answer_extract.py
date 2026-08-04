"""답 추출 모듈 — kaggle/02 노트북과 동일 로직 (백슬래시-안전 chr(92) 조립).

노트북과 서버 스크립트가 같은 추출기를 쓰도록 분리. 로직 수정 시 양쪽 모두 갱신할 것.
"""
import re

BS = chr(92)  # 백슬래시

NUM_PAT = re.compile('-?[0-9][0-9,]*(?:[.][0-9]+)?')
BOXED_PAT = re.compile('boxed *{((?:[^{}]|{[^{}]*})*)}')
FRAC_PAT = re.compile('frac *{ *(-?[0-9]+) *} *{ *(-?[0-9]+) *}')


def parse_int(s):
    s = ''.join(ch for ch in s if not ch.isspace())
    s = s.replace(BS, '')
    for junk in ',$!;:':
        s = s.replace(junk, '')
    s = s.strip('.')
    try:
        return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
        if abs(f - round(f)) < 1e-6:
            return int(round(f))
    except (ValueError, OverflowError):
        pass
    return None


def extract_answer(text):
    for cand in reversed(BOXED_PAT.findall(text)):
        m = FRAC_PAT.search(cand)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b != 0 and a % b == 0:
                return a // b
        val = parse_int(cand)
        if val is not None:
            return val
        for c in reversed(NUM_PAT.findall(cand)):
            val = parse_int(c)
            if val is not None:
                return val
    for cand in reversed(NUM_PAT.findall(text)):
        val = parse_int(cand)
        if val is not None:
            return val
    return None
