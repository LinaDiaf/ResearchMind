import uuid

from researchmind.retrieval.fusion import weighted_rrf


def test_weighted_rrf():
    a, b, c = (uuid.uuid4() for _ in range(3))
    dense = [(a, 0.9), (b, 0.8)]
    lexical = [(b, 12.0), (c, 9.0)]
    fused = [d for d, _ in weighted_rrf([(1.0, dense), (1.0, lexical)])]
    assert fused[0] == b  # appears in both lists
    lex_heavy = [d for d, _ in weighted_rrf([(0.2, dense), (3.0, lexical)])]
    assert lex_heavy.index(c) < lex_heavy.index(a)
    assert [d for d, _ in weighted_rrf([(0.0, dense), (1.0, lexical)])] == [b, c]
