"""Offline graph-path tests; model/runtime dependencies are mocked at import boundaries."""
import importlib.util
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]


def load_retriever():
    # Import the complete production module without loading a model, GPU runtime,
    # FAISS index, configuration, or API client. Restore sys.modules afterwards
    # so these mocks cannot leak into other tests.
    dependencies = {
        name: MagicMock()
        for name in (
            'faiss', 'numpy', 'spacy', 'torch', 'torch.nn',
            'torch.nn.functional', 'sentence_transformers',
            'models.retriever.faiss_filter', 'utils.call_llm_api',
            'utils.logger', 'config',
        )
    }
    dependencies['utils.logger'].logger = logging.getLogger(__name__)
    with patch.dict(sys.modules, dependencies):
        spec = importlib.util.spec_from_file_location(
            '_path_test_retriever',
            ROOT / 'models/retriever/enhanced_kt_retriever.py',
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        graph_spec = importlib.util.spec_from_file_location(
            '_path_test_graph_processor', ROOT / 'utils/graph_processor.py',
        )
        graph_module = importlib.util.module_from_spec(graph_spec)
        graph_spec.loader.exec_module(graph_module)
    return module.KTRetriever, graph_module.load_graph_from_json


KTRetriever, load_graph_from_json = load_retriever()


class FakeVector:
    def cpu(self):
        return self

    def numpy(self):
        return self


class PathRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.retriever = KTRetriever.__new__(KTRetriever)
        self.retriever.graph = nx.MultiDiGraph()

    def node(self, node_id, name):
        self.retriever.graph.add_node(
            node_id, label='entity', properties={'name': name, 'chunk id': name},
        )

    def edge(self, head, relation, tail):
        self.retriever.graph.add_edge(head, tail, relation=relation)

    def search(self, starts=('a',), keywords=('paris',), depth=2):
        return self.retriever._path_based_search(list(starts), list(keywords), depth)

    def chain(self):
        for node_id, name in [('a', 'Alice'), ('b', 'Acme'), ('c', 'Paris')]:
            self.node(node_id, name)
        self.edge('a', 'works_at', 'b')
        self.edge('b', 'located_in', 'c')

    def test_empty_graph_and_start_nodes(self):
        self.assertEqual(self.search(starts=()), [])

    def test_empty_keywords(self):
        self.chain()
        self.assertEqual(self.search(keywords=()), [])

    def test_no_matching_keyword(self):
        self.chain()
        self.assertEqual(self.search(keywords=('london',)), [])

    def test_disconnected_target(self):
        self.chain()
        self.retriever.graph.remove_edge('b', 'c')
        self.assertEqual(self.search(), [])

    def test_direct_path_uses_node_ids(self):
        self.chain()
        self.assertEqual(self.search(keywords=('acme',)), [('a', 'works_at', 'b')])

    def test_two_hop_path(self):
        self.chain()
        self.assertEqual(self.search(), [('a', 'works_at', 'b'), ('b', 'located_in', 'c')])

    def test_parallel_relations_are_preserved(self):
        self.chain()
        self.edge('b', 'founded_in', 'c')
        self.assertCountEqual(self.search(), [
            ('a', 'works_at', 'b'), ('b', 'located_in', 'c'), ('b', 'founded_in', 'c'),
        ])

    def test_reverse_edges_are_not_traversed_or_emitted(self):
        self.chain()
        self.edge('c', 'hosts', 'b')
        self.assertEqual(self.search(starts=('c',), keywords=('alice',)), [])
        self.assertEqual(self.search(), [('a', 'works_at', 'b'), ('b', 'located_in', 'c')])

    def test_depth_limit(self):
        self.chain()
        self.assertEqual(self.search(depth=1), [])
        self.assertEqual(self.search(depth=0), [])
        self.assertEqual(len(self.search(depth=2)), 2)

    def test_cycle_terminates(self):
        self.chain()
        self.edge('b', 'employs', 'a')
        self.assertEqual(self.search(), [('a', 'works_at', 'b'), ('b', 'located_in', 'c')])

    def test_parallel_edge_without_relation_does_not_hide_valid_edge(self):
        self.node('a', 'Alice')
        self.node('b', 'Paris')
        self.retriever.graph.add_edge('a', 'b', description='no relation')
        self.edge('a', 'visited', 'b')
        self.assertEqual(self.search(), [('a', 'visited', 'b')])

    def test_loaded_graph_reaches_final_answer_context(self):
        def entity(name):
            return {'label': 'entity', 'properties': {'name': name, 'chunk id': name}}

        records = [
            {'start_node': entity(head), 'relation': relation, 'end_node': entity(tail)}
            for head, relation, tail in [
                ('Alice', 'works_at', 'Acme'),
                ('Acme', 'located_in', 'Paris'),
                ('Acme', 'founded_in', 'Paris'),
            ]
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.json'
            path.write_text(json.dumps(records), encoding='utf-8')
            self.retriever.graph = load_graph_from_json(str(path))
        self.assertIsInstance(self.retriever.graph, nx.MultiDiGraph)
        retriever = self.retriever
        retriever.config = None
        retriever.top_k = 5
        retriever.recall_paths = 2
        retriever.chunk2id = {name: f'Source for {name}' for name in ('Alice', 'Acme', 'Paris')}
        vector = FakeVector()
        retriever.faiss_retriever = SimpleNamespace(transform_vector=lambda value: value)
        # Deterministic model/search boundaries: the answer-bearing second hop
        # must come from graph traversal, not vector or chunk retrieval.
        with patch.multiple(
            retriever,
            _get_query_embedding=lambda _: vector,
            _execute_faiss_node_search=lambda *_: ['entity_0'],
            _execute_faiss_relation_search=lambda *_: [],
            _extract_query_keywords=lambda _: ['paris'],
            _keyword_based_node_search=lambda _: [],
            _batch_calculate_entity_similarities=lambda _, nodes: {node: 1.0 for node in nodes},
            _chunk_embedding_retrieval=lambda *_: {},
            _triple_only_retrieval=lambda _: {'scored_triples': []},
            _rerank_triples_by_relevance=lambda triples, _: [(*triple, 1.0) for triple in triples],
        ):
            context, _ = retriever.process_retrieval_results('Where is Alice employer located?')
        self.assertEqual(set(context['chunk_ids']), {'Alice', 'Acme', 'Paris'})
        self.assertIn('Source for Paris', context['chunk_contents'])
        self.assertTrue(any('located_in' in triple and 'Paris' in triple for triple in context['triples']))
        self.assertTrue(any('founded_in' in triple and 'Paris' in triple for triple in context['triples']))
        self.assertEqual(len(context['triples']), 3)


if __name__ == '__main__':
    unittest.main()
