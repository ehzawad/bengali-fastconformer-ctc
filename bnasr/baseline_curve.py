#!/usr/bin/env python3
"""Extract the baseline's TRUE validation curve from its TensorBoard event files.

Why this exists: council packets H..O carried a curve built by enumerating the
"New best score" lines in lightning_logs.txt and labelling event i as update
(i+1)*1000.  Those lines are EarlyStopping improvement events, not a 1,000-step
grid, so the resulting curve was the running-minimum envelope re-indexed onto
the wrong x-axis - understating the true update by up to 16,005 (84%).  The
event files hold the real per-validation series; this reads them directly.

Panel is data2/val_corpus.json (5,385 rows) = this recipe's panel D, greedy CTC,
NeMo-internal WER on raw text.  Global step is continuous across the crash and
resume, so version_1's steps are absolute updates.
"""
import glob, json, struct, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / 'experiments/fastconformer_ctc_bn_1kh'


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s
        if not c & 0x80:
            return r, i
        s += 7


def _records(path):
    data = Path(path).read_bytes()
    i, n = 0, len(data)
    while i + 12 <= n:
        (ln,) = struct.unpack('<Q', data[i:i + 8]); i += 12
        if i + ln + 4 > n:
            return
        yield data[i:i + ln]; i += ln + 4


def _event(rec):
    i, step, summ = 0, None, None
    while i < len(rec):
        key, i = _varint(rec, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(rec, i)
            if fn == 2:
                step = v
        elif wt == 1:
            i += 8
        elif wt == 2:
            ln, i = _varint(rec, i)
            if fn == 5:
                summ = rec[i:i + ln]
            i += ln
        elif wt == 5:
            i += 4
        else:
            break
    return step, summ


def _scalars(summ):
    out, i = [], 0
    while i < len(summ):
        key, i = _varint(summ, i)
        fn, wt = key >> 3, key & 7
        if wt == 2:
            ln, i = _varint(summ, i); val = summ[i:i + ln]; i += ln
            if fn != 1:
                continue
            tag = sv = None; j = 0
            while j < len(val):
                k2, j = _varint(val, j)
                f2, w2 = k2 >> 3, k2 & 7
                if w2 == 2:
                    l2, j = _varint(val, j)
                    if f2 == 1:
                        tag = val[j:j + l2].decode('utf8', 'replace')
                    j += l2
                elif w2 == 5:
                    sv = struct.unpack('<f', val[j:j + 4])[0]; j += 4
                elif w2 == 0:
                    _, j = _varint(val, j)
                elif w2 == 1:
                    j += 8
                else:
                    break
            if tag is not None:
                out.append((tag, sv))
        elif wt == 0:
            _, i = _varint(summ, i)
        elif wt == 5:
            i += 4
        elif wt == 1:
            i += 8
        else:
            break
    return out


def series(tags=('val_wer', 'learning_rate', 'epoch')):
    """{tag: [(global_step, value), ...]} merged over version_0 and version_1."""
    acc = {t: [] for t in tags}
    for vd in ('version_0', 'version_1'):
        for p in sorted(glob.glob(str(LOGDIR / vd / 'events.out.tfevents.*'))):
            for rec in _records(p):
                step, summ = _event(rec)
                if summ is None:
                    continue
                for tag, sv in _scalars(summ):
                    if tag in acc and sv is not None:
                        acc[tag].append((step, sv))
    for t in acc:
        acc[t].sort()
    return acc


def val_wer_by_update():
    """{update_rounded_to_1000: internal WER pct} - the TRUE curve."""
    s = series()['val_wer']
    return {int(round(st / 1000.0) * 1000): round(v * 100, 4) for st, v in s}


def main():
    s = series()
    lr = dict(s['learning_rate'])
    rows = []
    for st, v in s['val_wer']:
        near = min(lr, key=lambda k: abs(k - st)) if lr else None
        rows.append({'tb_step': st, 'update': int(round(st / 1000.0) * 1000),
                     'val_wer_pct': round(v * 100, 4),
                     'lr': lr.get(near)})
    best = min(rows, key=lambda r: r['val_wer_pct'])
    out = {'source': 'experiments/fastconformer_ctc_bn_1kh/version_{0,1}/events.out.tfevents.*',
           'tag': 'val_wer', 'panel': 'data2/val_corpus.json (5,385 rows) == this recipe panel D',
           'surface': 'NeMo-internal WER on raw text, greedy CTC',
           'n_validations': len(rows), 'best': best,
           'note': ('Councils H..O used a curve built from lightning_logs.txt "New best score" lines '
                    'labelled (i+1)*1000. Those are improvement events, not a 1,000-step grid: the '
                    'resulting series is the running-minimum envelope on a wrong x-axis, understating '
                    'the true update by up to 16,005 (84%). This file supersedes it.'),
           'curve': rows}
    dest = ROOT / 'experiments/phase1_val_curve.json'
    dest.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(f'wrote {dest} : {len(rows)} validations, best {best["val_wer_pct"]}% at update {best["update"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
