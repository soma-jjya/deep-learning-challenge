"""문제 텍스트 정화 함수 (exp21).

크롤링 과정에서 문제 본문에 섞여 들어간 잡음 패턴 4종을 제거한다.
효과가 확인되면 make_submission.py에도 이식할 목적으로 함수만 분리해둔다.

잡음 패턴:
  1. 말미의 "Translate the above text into English..." 지시문 — 원문 자체가 문제
     인데, 번역 파이프라인 잔재 프롬프트가 그대로 붙어 있음
  2. 마크다운 이미지 링크 ![](...)  — LLM은 이미지를 볼 수 없어 순수 잡음(토큰 낭비)
  3. [asy]...[/asy]  Asymptote 도형 코드 — 텍스트만으로 활용 못 하는 저수준 그림
     언어, 대부분 잡음
  4. 문두 문제 번호 "N. " — 원문제집 스캔본의 numbering 잔재
"""
import re

_TRANSLATE_RE = re.compile(r'\s*Translate the above text.*$', re.I | re.S)
_IMG_RE = re.compile(r'!\[[^\]]*\]\([^)]*\)')
_ASY_RE = re.compile(r'\[asy\].*?\[/asy\]', re.I | re.S)
_LEADNUM_RE = re.compile(r'^\s*\d+\.\s+')


def clean_question(text):
    if not isinstance(text, str):
        return text
    cleaned = text
    cleaned = _TRANSLATE_RE.sub('', cleaned)
    cleaned = _IMG_RE.sub('', cleaned)
    cleaned = _ASY_RE.sub('', cleaned)
    cleaned = _LEADNUM_RE.sub('', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def is_affected(text):
    """정화가 실제로 텍스트를 바꾸는지 여부 (영향받은 문항 집계용)."""
    if not isinstance(text, str):
        return False
    return clean_question(text) != text
