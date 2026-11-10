"""TS-Haystack's answer scoring, ported from its evaluate_answer
(src/datasets/capture24_haystack/utils/answer_evaluation.py, CC BY-NC 4.0) with the LTAF settings:
time ranges right at IoU >= 0.25, single timestamps within 0.5 s, yes/no and integers after
normalisation, categories as normalised strings with the label found inside the prediction."""

import re
from datetime import datetime, timedelta

TIME = r"(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?(?:\s*[AP]M)?)"
IOU = 0.25
TOLERANCE_S = 0.5


def parse_time(s: str) -> datetime:
    s, ms = s.strip(), 0
    if "." in s:
        base, frac = s.rsplit(".", 1)
        if " " in frac:
            frac, ampm = frac.split(" ", 1)
            base = f"{base} {ampm}"
        s, ms = base, int(frac)
    for fmt in ["%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"]:
        try:
            return datetime.strptime(s, fmt) + timedelta(milliseconds=ms)
        except ValueError:
            continue
    raise ValueError(s)


def time_range(s: str) -> tuple[datetime, datetime] | None:
    m = re.findall(TIME, s, re.IGNORECASE)
    if len(m) < 2:
        return None
    try:
        return parse_time(m[0]), parse_time(m[-1])
    except ValueError:
        return None


def iou(p: tuple[datetime, datetime], g: tuple[datetime, datetime]) -> float:
    (ps, pe), (gs, ge) = p, g
    pe = pe + timedelta(days=1) if pe < ps else pe
    ge = ge + timedelta(days=1) if ge < gs else ge
    lo, hi = max(ps, gs), min(pe, ge)
    if lo >= hi:
        return 0.0
    inter = (hi - lo).total_seconds()
    union = (pe - ps).total_seconds() + (ge - gs).total_seconds() - inter
    return inter / union if union > 0 else 0.0


def boolean(s: str) -> str | None:
    s = s.strip().lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    head = s[:30]
    if "it does" in head and "doesn't" not in head and "does not" not in head:
        return "yes"
    if "doesn't" in head or "does not" in head:
        return "no"
    return None


def integer(s: str) -> int | None:
    m = re.search(r"\d+", str(s).strip())
    return int(m.group()) if m else None


def no_answer_match(gt: str, pred: str, g, p) -> bool:
    """Their fallback when a side does not parse: both starting with "no" is right."""
    if g is None and p is None:
        if gt.strip().lower().startswith("no") and pred.strip().lower().startswith("no"):
            return True
        return re.sub(r"\s+", " ", gt.strip().lower()) == re.sub(r"\s+", " ", pred.strip().lower())
    if gt.strip().lower().startswith("no") and pred.strip().lower().startswith("no"):
        return True
    return False


def correct(gt: str, pred: str, answer_type: str) -> bool:
    if answer_type == "boolean":
        g, p = boolean(gt), boolean(pred)
        return g == p if g and p else gt.strip().lower() == pred.strip().lower()
    if answer_type == "integer":
        g, p = integer(gt), integer(pred)
        return g == p if g is not None and p is not None else gt.strip() == pred.strip()
    if answer_type == "time_range":
        g, p = time_range(gt), time_range(pred)
        if g is not None and p is not None:
            return iou(p, g) >= IOU
        return no_answer_match(gt, pred, g, p)
    if answer_type == "timestamp":
        gm, pm = re.findall(TIME, gt, re.IGNORECASE), re.findall(TIME, pred, re.IGNORECASE)
        g = p = None
        try:
            g = parse_time(gm[0]) if gm else None
        except ValueError:
            pass
        try:
            p = parse_time(pm[0]) if pm else None
        except ValueError:
            pass
        if g is not None and p is not None:
            return abs((p - g).total_seconds()) <= TOLERANCE_S
        return no_answer_match(gt, pred, g, p)
    g = re.sub(r"[.,;:!?]+$", "", gt.strip().lower())
    p = re.sub(r"[.,;:!?]+$", "", pred.strip().lower())
    # Theirs counts an empty prediction as right, since "" is inside every label.
    if not p:
        return False
    if g == p:
        return True
    gs, ps = g.replace("-", " "), p.replace("-", " ")
    if gs in ps or ps in gs:
        return True
    gt_tok, p_tok = re.findall(r"[A-Za-z0-9]+", gs), re.findall(r"[A-Za-z0-9]+", ps)
    return bool(gt_tok and p_tok and gt_tok[-1] == p_tok[-1])
