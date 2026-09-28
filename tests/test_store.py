from skilled_proposer.store import SeenStore, tag_records, text_hash


def rec(feedback):
    return {"Inputs": {"q": feedback}, "Generated Outputs": {"a": "x"}, "Feedback": feedback}


def build():
    store = SeenStore()
    store.add("p", "seed", tag_records([rec("s1")], component="p", iteration=1, instruction="seed"))
    store.link("p", "seed", "A")
    store.add("p", "A", tag_records([rec("a1")], component="p", iteration=2, instruction="A"))
    store.link("p", "A", "B")
    store.link("p", "seed", "C")  # a sibling of A
    store.add("p", "C", tag_records([rec("c1")], component="p", iteration=3, instruction="C"))
    store.add("q", "other", tag_records([rec("q1")], component="q", iteration=3, instruction="other"))
    return store


def tags_by_feedback(view):
    return {r["Feedback"]: r["tags"] for r in view["records"]}


def test_tag_records_adds_provenance_and_json_types():
    [r] = tag_records([{"Inputs": {"q": ("a",)}, "Feedback": "wrong"}],
                      component="p", iteration=3, instruction="Do X.")
    assert r["Inputs"] == {"q": ["a"]}
    assert r["tags"] == {"iteration": 3, "component": "p", "instruction": text_hash("Do X.")}


def test_view_tags_relations_from_lineage():
    view = build().view("p", "B", iteration=4, seed=0)
    tags = tags_by_feedback(view)
    assert set(tags) == {"s1", "a1", "c1"}
    assert (tags["a1"]["relation"], tags["a1"]["edits_back"]) == ("ancestor", 1)
    assert (tags["s1"]["relation"], tags["s1"]["edits_back"]) == ("ancestor", 2)
    assert (tags["c1"]["relation"], tags["c1"]["edits_back"]) == ("other_branch", None)
    assert view["instructions"][text_hash("A")] == {"text": "A", "relation": "ancestor", "edits_back": 1}


def test_view_marks_records_from_the_current_text():
    tags = tags_by_feedback(build().view("p", "A", iteration=4, seed=0))
    assert (tags["a1"]["relation"], tags["a1"]["edits_back"]) == ("current", 0)
    assert tags["s1"]["relation"] == "ancestor"


def test_view_excludes_other_components_and_later_iterations():
    view = build().view("p", "B", iteration=3, seed=0)
    assert {r["Feedback"] for r in view["records"]} == {"s1", "a1"}


def test_view_shuffle_is_seeded():
    store = SeenStore()
    store.add("p", "seed", tag_records([rec(str(i)) for i in range(20)],
                                       component="p", iteration=1, instruction="seed"))

    def order(seed, iteration):
        return [r["Feedback"] for r in store.view("p", "seed", iteration, seed)["records"]]

    assert order(0, 2) == order(0, 2)
    assert order(0, 2) != order(0, 3)
    assert sorted(order(0, 2)) == sorted(str(i) for i in range(20))


def test_lineage_cycle_terminates():
    store = SeenStore()
    store.link("p", "A", "B")
    store.link("p", "B", "A")
    assert store.view("p", "A", 1, 0) == {"instructions": {}, "records": []}


def test_save_and_load(tmp_path):
    store = build()
    path = tmp_path / "seen.json"
    store.save(path)
    back = SeenStore.load(path)
    assert back.view("p", "B", 4, 0) == store.view("p", "B", 4, 0)
    assert SeenStore.load(tmp_path / "missing.json").records == {}
