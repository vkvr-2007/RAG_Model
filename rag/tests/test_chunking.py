from app.chunking import chunk_text


def test_all_chunkers_produce_text():
    text = "पहला वाक्य है। दूसरा वाक्य बहुत उपयोगी है। तीसरा वाक्य भी संदर्भ रखता है।"
    for strategy in ("passage", "sentence", "recursive"):
        assert chunk_text(text, strategy, chunk_size=5, overlap=2)


def test_sentence_overlap():
    chunks = chunk_text("एक दो तीन। चार पांच छह। सात आठ नौ।", "sentence", chunk_size=4, overlap=2)
    assert len(chunks) > 1 and "तीन" in chunks[1]
