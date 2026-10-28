from issue_triage.storage import HashEmbeddings


def test_hash_embeddings_are_repeatable_and_normalized() -> None:
    embeddings = HashEmbeddings(dimensions=128)
    first = embeddings.embed_query("database connection timeout")
    second = embeddings.embed_query("database connection timeout")
    norm = sum(value * value for value in first) ** 0.5

    assert first == second
    assert abs(norm - 1.0) < 1e-9


def test_empty_hash_embedding_is_zero_vector() -> None:
    vector = HashEmbeddings(dimensions=16).embed_query("   ")

    assert vector == [0.0] * 16
