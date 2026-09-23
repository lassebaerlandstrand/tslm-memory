import numpy as np
import pandas as pd
import pytest

from tslm_context.mimic import bucket, earlier, labels_of, next_pairs


def test_qualifier_applies_to_its_own_clause_only():
    assert labels_of(["Sinus tachycardia with borderline first degree A-V block"]) == (
        ["sinus tachycardia"],
        ["first degree av block"],
    )
    assert labels_of(["Probable atrial fibrillation"]) == ([], ["atrial fibrillation or flutter"])
    assert labels_of(["Possible anterior infarct - age undetermined"]) == ([], ["infarct pattern"])


def test_definite_mention_wins_over_qualified_one():
    got = labels_of(["Possible atrial fibrillation", "Atrial fibrillation"])
    assert got == (["atrial fibrillation or flutter"], [])


def test_findings_are_recognised_once():
    assert labels_of(["Atrial flutter with 2:1 A-V conduction"])[0] == [
        "atrial fibrillation or flutter"
    ]
    assert "prolonged qt" in labels_of(["Prolonged QT interval"])[0]
    assert "st-t changes suggesting ischemia" in labels_of(["ST-T changes suggest ischemia"])[0]
    assert labels_of(["Sinus rhythm", "Sinus rhythm"]) == (["sinus rhythm"], [])


def test_bucket_is_stable_in_range_and_rejects_floats():
    assert bucket(12345678) == bucket(np.int64(12345678))
    assert {bucket(i) for i in range(1000)} == set(range(10))
    with pytest.raises(AssertionError):
        bucket(12345678.0)


def table(rows: list[tuple[int, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["subject_id", "ecg_time"])
    df["ecg_time"] = pd.to_datetime(df.ecg_time)
    return df.sort_values(["subject_id", "ecg_time"]).reset_index(drop=True)


ECGS = table(
    [
        (1, "2150-01-01 08:00"),
        (1, "2150-01-01 08:30"),
        (1, "2150-01-01 09:00"),
        (1, "2150-01-01 09:59"),
        (1, "2150-01-01 11:00"),
        (1, "2150-01-01 11:00"),  # duplicate time
        (2, "2150-01-01 07:00"),  # another subject whose shifted dates overlap subject 1's
        (2, "2150-06-01 07:00"),
    ]
)


def test_earlier_keeps_subject_and_one_hour_spacing():
    rows = np.array([4, 7, 6])
    hist = earlier(ECGS, rows, 3)
    # 11:00 -> 09:59 (61 min before) -> 08:30 (09:00 is only 59 min before 09:59) -> none.
    assert hist.tolist() == [[3, 1, -1], [6, -1, -1], [-1, -1, -1]]
    t = ECGS.ecg_time.to_numpy()
    for row, h in zip(rows, hist, strict=True):
        prev = t[row]
        for j in h[h >= 0]:
            assert ECGS.subject_id[j] == ECGS.subject_id[row]
            assert prev - t[j] >= np.timedelta64(60, "m")
            prev = t[j]


def test_earlier_and_next_pairs_reject_a_gapped_index():
    with pytest.raises(AssertionError):
        earlier(ECGS.drop(index=2), np.array([0]), 1)
    with pytest.raises(AssertionError):
        next_pairs(ECGS.drop(index=2))


def test_next_pairs_respects_gap_and_subject():
    got = dict(zip(*next_pairs(ECGS)[["cur", "nxt"]].to_numpy().T, strict=True))
    assert got == {0: 2, 1: 3, 2: 4, 3: 4, 6: 7}
    got = dict(zip(*next_pairs(ECGS, max_d=100)[["cur", "nxt"]].to_numpy().T, strict=True))
    assert 6 not in got
