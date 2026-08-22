import asyncio
from app.generation import Generator
from app.schemas import GeneratedAnswer
from app.service import RAGService


class FakeRetriever:
    def __init__(self, score=0.9): self.score = score
    def search_with_timings(self, query, k):
        data = {"query_id":"1", "chunk_id":"c1", "text":"दिल्ली भारत की राजधानी है।", "language":"hi", "chunking_strategy":"passage"}
        return ([type("C", (), {"metadata": data, "score": self.score})()], self.score, {"embedding_ms":0, "vector_retrieval_ms":0, "bm25_ms":0, "fusion_ms":0})


def test_low_confidence_is_not_grounded():
    result = asyncio.run(RAGService(FakeRetriever(0.01), Generator(None,None,None,1), 1, .35).query("क्या है"))
    assert not result.grounded


def test_ungrounded_generator_response_is_hidden():
    result = asyncio.run(RAGService(FakeRetriever(), Generator(None,None,None,1), 1, .35, strict_extractive=False).query("राजधानी क्या है"))
    assert not result.grounded and not result.sources


class GroundedGenerator:
    async def answer(self, query, context):
        return GeneratedAnswer(answer="दिल्ली भारत की राजधानी है।", grounded=True)


def test_grounded_response_has_sources():
    result = asyncio.run(RAGService(FakeRetriever(), GroundedGenerator(), 1, .35).query("राजधानी क्या है"))
    assert result.grounded and result.sources and result.answer


class TopChunkRetriever:
    def __init__(self, text: str, score=0.9):
        self.text = text
        self.score = score

    def search_with_timings(self, query, k):
        data = {"query_id":"1", "chunk_id":"c1", "text": self.text, "language":"hi", "chunking_strategy":"passage"}
        return ([type("C", (), {"metadata": data, "score": self.score})()], self.score, {"embedding_ms":0, "vector_retrieval_ms":0, "bm25_ms":0, "fusion_ms":0})


def test_top_chunk_must_match_query_entity():
    retriever = TopChunkRetriever("जर्मनी की राजधानी बर्लिन है।")
    result = asyncio.run(RAGService(retriever, Generator(None,None,None,1), 1, .35).query("चीन गणराज्य की राजधानी क्या है?"))
    assert not result.grounded and not result.sources


def test_alabama_regression_rejected_without_entity_match():
    retriever = TopChunkRetriever("लिटिल रॉक अर्कांसस की राजधानी है।")
    result = asyncio.run(RAGService(retriever, Generator(None,None,None,1), 1, .35).query("अलबामा की राजधानी क्या है?"))
    assert not result.grounded and not result.sources


def test_matching_entity_chunk_is_grounded():
    retriever = TopChunkRetriever("बीजिंग चीन गणराज्य की राजधानी है।")
    result = asyncio.run(RAGService(retriever, Generator(None,None,None,1), 1, .35).query("चीन गणराज्य की राजधानी क्या है?"))
    assert result.grounded and result.answer
