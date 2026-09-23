from tslm_context.mimic import bucket, labels_of


def test_qualified_statements_do_not_count():
    assert labels_of(["Incomplete right bundle branch block"]) == []
    assert labels_of(["Borderline prolonged QT interval"]) == []
    assert labels_of(["Possible anterior infarct - age undetermined"]) == []


def test_findings_are_recognised():
    assert labels_of(["Atrial flutter with 2:1 A-V conduction"]) == [
        "atrial fibrillation or flutter"
    ]
    assert labels_of(["Sinus tachycardia"]) == ["sinus tachycardia"]
    assert "prolonged qt" in labels_of(["Prolonged QT interval"])
    assert "st-t changes suggesting ischemia" in labels_of(["ST-T changes suggest ischemia"])


def test_labels_are_not_repeated():
    assert labels_of(["Sinus rhythm", "Sinus rhythm"]) == ["sinus rhythm"]


def test_bucket_is_stable_and_in_range():
    assert bucket(12345678) == bucket(12345678)
    assert {bucket(i) for i in range(1000)} == set(range(10))
