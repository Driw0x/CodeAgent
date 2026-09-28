from app.rag.citations import extract_citations, is_abstention, validate_citations


def test_extract_citations():
    assert extract_citations("Réponse [S1] puis [S3].") == [1, 3]


def test_extract_citations_empty():
    assert extract_citations("Réponse sans source.") == []


def test_abstention():
    assert is_abstention("Je ne peux pas le déterminer à partir des sources disponibles.")


def test_partial_abstention():
    assert is_abstention("La première partie est supportée [S1]. Je ne peux pas déterminer le reste à partir des sources disponibles.")


def test_not_abstention():
    assert not is_abstention("Cette fonction charge l'index [S1].")


def test_valid_citation():
    assert validate_citations("Cette fonction charge l'index [S1].", 2)


def test_multiple_valid_citations():
    assert validate_citations("La première partie vient de [S1] et la seconde de [S2].", 2)


def test_invalid_citation():
    assert not validate_citations("Cette fonction charge l'index [S3].", 2)


def test_missing_citation():
    assert not validate_citations("Cette fonction charge l'index.", 2)


def test_abstention_without_citation():
    assert validate_citations("Je ne peux pas le déterminer à partir des sources disponibles.", 2)


def test_partial_abstention_with_valid_citation():
    assert validate_citations("Cette partie est supportée [S1]. Je ne peux pas déterminer le reste à partir des sources disponibles.", 2)


def test_abstention_with_invalid_citation():
    assert not validate_citations("Je ne peux pas le déterminer à partir des sources disponibles [S3].", 2)