"""평가 정답 생성 전용 스크립트 — 학습·생성 짝·가중치 선택에 사용하지 않음.

원본 데이터에는 5자리 이상 영어↔한국어 정답 쌍이 거의 없고 9·10자리는 하나도 없어서,
num2text.ipynb 안의 test 평가로는 긴 숫자의 영어↔한국어 번역을 측정할 수 없습니다.
이 스크립트는 학습이 모두 끝난 뒤, test에 배정된 숫자에 대해서만 다음을 합니다.

1. 수 읽기 규칙이 원본 영어 100,000개·한국어 100,000개 표현과 모두 일치하는지 먼저 확인합니다.
2. test 숫자의 영어·한국어 표현을 준비합니다. 원본에 있으면 원본을, 없으면 규칙으로 만든 표현을 씁니다.
3. 저장된 세 모델로 영어→한국어, 한국어→영어를 두 방식으로 번역해 완전일치율을 자릿수별로 셉니다.
   - direct: 모델 하나로 바로 번역
   - pivot : 같은 모델로 숫자를 거쳐 번역 (영어→숫자→한국어)

규칙은 정답지를 만드는 데만 쓰고, 모델의 번역에는 쓰지 않습니다. 숫자는 정수로 바꾸지 않고
문자열 그대로 다룹니다.

실행: 이 폴더(practice01/doyeon)에서  python extra_eval.py
결과: results/extra_eval_metrics.csv, results/reference_validation.json
"""
import csv, json, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 노트북의 구현 셀(첫 코드 셀)을 그대로 불러와 같은 모델·번역·채점 함수를 씁니다.
_nb = json.loads((HERE / 'num2text.ipynb').read_text(encoding='utf-8'))
_impl = next(c for c in _nb['cells'] if c['cell_type'] == 'code')
NB = {'__name__': 'num2text_impl'}
exec(''.join(_impl['source']), NB)
NB['ROOT'] = HERE

# ---- 수 읽기 규칙 (정답지 생성 전용, 문자열 처리) ----
ONES = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
        'seventeen', 'eighteen', 'nineteen']
TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
DIGIT = '0123456789'

def _en_three(s):  # s: 1~3자리 문자열
    s = s.rjust(3, '0'); h, t, o = DIGIT.index(s[0]), DIGIT.index(s[1]), DIGIT.index(s[2])
    words = []
    if h: words += [ONES[h], 'hundred']
    if t >= 2:
        words.append(TENS[t])
        if o: words.append(ONES[o])
    elif t * 10 + o:
        words.append(ONES[t * 10 + o])
    return words

def english(s):
    s = s.lstrip('0')
    if not s: return 'zero'
    groups = []
    while s: groups.insert(0, s[-3:]); s = s[:-3]
    scales = ['', 'thousand', 'million', 'billion']
    words = []
    for i, g in enumerate(groups):
        part = _en_three(g)
        if part:
            words += part
            scale = scales[len(groups) - 1 - i]
            if scale: words.append(scale)
    return ' '.join(words)

KO = '영일이삼사오육칠팔구'

def _ko_four(s):  # s: 1~4자리 문자열
    s = s.rjust(4, '0'); out = ''
    for ch, unit in zip(s, ['천', '백', '십', '']):
        d = DIGIT.index(ch)
        if d: out += ('' if d == 1 and unit else KO[d]) + unit
    return out

def korean(s):
    s = s.lstrip('0')
    if not s: return '영'
    groups = []
    while s: groups.insert(0, s[-4:]); s = s[:-4]
    units = ['', '만', '억']
    parts = []
    for i, g in enumerate(groups):
        body = _ko_four(g)
        if body:
            unit = units[len(groups) - 1 - i]
            if unit == '만' and body == '일': body = ''
            parts.append(body + unit)
    return ' '.join(parts)

def main():
    records = NB['read_data']()
    # 1) 규칙이 원본 표현과 모두 같은지 확인
    mismatch = [(n, lang) for n, r in records.items() for lang, text in r.items()
                if lang in ('en', 'ko') and (english(n) if lang == 'en' else korean(n)) != text]
    checked = {lang: sum(lang in r for r in records.values()) for lang in ('en', 'ko')}
    NB['save_json'](HERE / 'results/reference_validation.json',
                    dict(checked=checked, mismatch_count=len(mismatch), examples=mismatch[:5]))
    assert not mismatch, f'규칙이 원본과 다릅니다: {mismatch[:5]}'
    print('규칙 확인:', checked, '불일치 0개')

    # 2) test 숫자의 영어·한국어 표현 (원본 우선, 없으면 규칙)
    split = json.loads((HERE / 'results/split_ids.json').read_text(encoding='utf-8'))
    test = sorted(split['test'], key=lambda n: (len(n), n))
    text = {n: {'en': records[n].get('en') or english(n),
                'ko': records[n].get('ko') or korean(n)} for n in test}

    def run(model, codec, items):
        ps = [dict(number=n, source=s, target=None, src=a, tgt=b, direction=a + '2' + b, weight=1.)
              for n, s, a, b in items]
        return [r['prediction'] if r['eos'] else None for r in NB['predictions'](model, ps, codec)]

    rows = []
    for name in ['baseline', 'baseline_continued', 'pseudo_augmented']:
        model, codec, _ = NB['load_model'](HERE / f'{name}.pt')
        for a, b in [('en', 'ko'), ('ko', 'en')]:
            direct = run(model, codec, [(n, text[n][a], a, b) for n in test])
            middle = run(model, codec, [(n, text[n][a], a, 'num') for n in test])
            pivot = run(model, codec, [(n, m or '', 'num', b) for n, m in zip(test, middle)])
            for method, preds in [('direct', direct), ('pivot', pivot)]:
                by = collections.defaultdict(list)
                for n, p in zip(test, preds):
                    ok = p == text[n][b]
                    by[str(len(n))].append(ok); by['all'].append(ok)
                for digits, oks in by.items():
                    rows.append(dict(model=name, method=method, direction=a + '2' + b, digits=digits,
                                     n=len(oks), correct=sum(oks), exact_match=sum(oks) / len(oks)))
            print(name, a + '2' + b, 'direct', sum(p == text[n][b] for n, p in zip(test, direct)),
                  'pivot', sum(p == text[n][b] for n, p in zip(test, pivot)), '/', len(test), flush=True)

    with (HERE / 'results/extra_eval_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('저장: results/extra_eval_metrics.csv')

if __name__ == '__main__':
    main()
