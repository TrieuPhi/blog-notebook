"""
Unit tests for Personal Knowledge Graph Engine and NetworkX 3.7.
Covers the implementation in personal-knowledge-graph.md.
"""
from __future__ import annotations

import re
import unittest
from dataclasses import dataclass, field

import networkx as nx

# ==============================================================================
# TEST ENGINE IMPLEMENTATION CO-LOCATED FOR DETERMINISTIC RUNTIME
# ==============================================================================

@dataclass
class NoteDocument:
    slug: str
    title: str
    content: str
    category: str = "General"
    tags: list[str] = field(default_factory=list)
    word_count: int = 0


class PersonalKnowledgeGraph:
    """
    Knowledge Graph engine powered by NetworkX.
    Extracts wikilinks, builds directed graph, calculates PageRank and centralities.
    """

    WIKILINK_PATTERN = re.compile(r"\[\[([^\[\]\|]+)(?:\|[^\]]+)?\]\]")

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._title_to_slug: dict[str, str] = {}
        self._documents: dict[str, NoteDocument] = {}

    def extract_wikilinks(self, markdown_text: str) -> list[str]:
        if not markdown_text:
            return []
        matches = self.WIKILINK_PATTERN.findall(markdown_text)
        return [match.strip() for match in matches if match.strip()]

    def add_document(self, doc: NoteDocument) -> None:
        self._documents[doc.slug] = doc
        self._title_to_slug[doc.title.strip().lower()] = doc.slug
        self.graph.add_node(
            doc.slug,
            title=doc.title,
            category=doc.category,
            tags=doc.tags,
            word_count=doc.word_count,
        )

    def build_graph(self) -> None:
        for source_slug, doc in self._documents.items():
            targets = self.extract_wikilinks(doc.content)
            for target_title in targets:
                target_slug = self._title_to_slug.get(target_title.lower())
                if target_slug:
                    self.graph.add_edge(source_slug, target_slug)
                else:
                    ghost_slug = target_title.lower().replace(" ", "-")
                    if not self.graph.has_node(ghost_slug):
                        self.graph.add_node(ghost_slug, title=target_title, category="Uncreated", tags=[], word_count=0)
                    self.graph.add_edge(source_slug, ghost_slug)

    def compute_centrality_metrics(self, alpha: float = 0.85) -> dict[str, dict[str, float]]:
        if len(self.graph) == 0:
            return {}

        in_degree = nx.in_degree_centrality(self.graph)
        out_degree = nx.out_degree_centrality(self.graph)
        pagerank = nx.pagerank(self.graph, alpha=alpha)
        betweenness = nx.betweenness_centrality(self.graph)

        metrics: dict[str, dict[str, float]] = {}
        for node in self.graph.nodes():
            metrics[node] = {
                "in_degree": in_degree.get(node, 0.0),
                "out_degree": out_degree.get(node, 0.0),
                "pagerank": pagerank.get(node, 0.0),
                "betweenness": betweenness.get(node, 0.0),
            }
        return metrics

    def get_backlinks(self, target_identifier: str) -> list[str]:
        target_slug = self._title_to_slug.get(target_identifier.strip().lower(), target_identifier)
        if not self.graph.has_node(target_slug):
            return []
        return list(self.graph.predecessors(target_slug))

    def find_shortest_path(self, source_identifier: str, target_identifier: str) -> list[str] | None:
        s_slug = self._title_to_slug.get(source_identifier.strip().lower(), source_identifier)
        t_slug = self._title_to_slug.get(target_identifier.strip().lower(), target_identifier)

        if not self.graph.has_node(s_slug) or not self.graph.has_node(t_slug):
            return None

        try:
            return nx.shortest_path(self.graph, source=s_slug, target=t_slug)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def detect_communities(self) -> list[set[str]]:
        if len(self.graph) < 2:
            return [set(self.graph.nodes())]

        undirected = self.graph.to_undirected()
        communities = list(nx.community.greedy_modularity_communities(undirected))
        return [set(comm) for comm in communities]


# ==============================================================================
# UNIT TEST CASES
# ==============================================================================

class TestKnowledgeGraph(unittest.TestCase):
    """Test suite for wikilink extraction, NetworkX graph creation, and graph algorithms."""

    def setUp(self):
        self.pkg = PersonalKnowledgeGraph()

    def test_wikilink_regex_extraction(self):
        text = (
            "This note references [[Machine Learning]] and also a piped link "
            "like [[Deep Learning|Hoc Sau]]. Multiple [[Python]] [[OOP]] links."
        )
        links = self.pkg.extract_wikilinks(text)
        self.assertEqual(links, ["Machine Learning", "Deep Learning", "Python", "OOP"])

    def test_wikilink_regex_empty_and_edge_cases(self):
        self.assertEqual(self.pkg.extract_wikilinks(""), [])
        self.assertEqual(self.pkg.extract_wikilinks("Plain text with no links [brackets]"), [])
        self.assertEqual(self.pkg.extract_wikilinks("[[]] and [[  ]]"), [])

    def test_graph_construction_and_metadata(self):
        doc1 = NoteDocument(
            slug="note-1",
            title="Note One",
            content="Referencing [[Note Two]] directly.",
            category="AI",
            tags=["AI", "ML"],
            word_count=500,
        )
        doc2 = NoteDocument(
            slug="note-2",
            title="Note Two",
            content="Terminal note with no links.",
            category="Backend",
            tags=["Python"],
            word_count=300,
        )

        self.pkg.add_document(doc1)
        self.pkg.add_document(doc2)
        self.pkg.build_graph()

        self.assertEqual(self.pkg.graph.number_of_nodes(), 2)
        self.assertEqual(self.pkg.graph.number_of_edges(), 1)
        self.assertTrue(self.pkg.graph.has_edge("note-1", "note-2"))

        # Verify node attributes
        node1_data = self.pkg.graph.nodes["note-1"]
        self.assertEqual(node1_data["title"], "Note One")
        self.assertEqual(node1_data["category"], "AI")
        self.assertEqual(node1_data["tags"], ["AI", "ML"])
        self.assertEqual(node1_data["word_count"], 500)

    def test_pagerank_and_degree_centrality(self):
        """
        Creates a hub topology where multiple notes point to a central Hub note.
        Hub note should have the highest In-Degree and highest PageRank.
        """
        notes = [
            NoteDocument("hub", "Central Hub", "Points to nothing."),
            NoteDocument("page-a", "Page A", "References [[Central Hub]]."),
            NoteDocument("page-b", "Page B", "References [[Central Hub]]."),
            NoteDocument("page-c", "Page C", "References [[Central Hub]] and [[Page A]]."),
        ]
        for n in notes:
            self.pkg.add_document(n)
        self.pkg.build_graph()

        metrics = self.pkg.compute_centrality_metrics(alpha=0.85)
        self.assertIn("hub", metrics)

        # PageRank sum across all nodes must equal 1.0 (within float precision)
        total_pr = sum(m["pagerank"] for m in metrics.values())
        self.assertAlmostEqual(total_pr, 1.0, places=4)

        # Central Hub must have highest In-degree and highest PageRank
        hub_pr = metrics["hub"]["pagerank"]
        for node, m in metrics.items():
            if node != "hub":
                self.assertGreaterEqual(hub_pr, m["pagerank"])

        self.assertEqual(metrics["hub"]["in_degree"], 1.0)  # All 3 other nodes point to hub (3/3 = 1.0)
        self.assertEqual(metrics["hub"]["out_degree"], 0.0)

    def test_betweenness_centrality_bridge_node(self):
        """
        Cluster 1: A <-> B
        Bridge: C (links A and D)
        Cluster 2: D <-> E
        Node C should have the highest Betweenness Centrality.
        """
        notes = [
            NoteDocument("a", "Note A", "[[Note B]] and [[Note C]]"),
            NoteDocument("b", "Note B", "[[Note A]]"),
            NoteDocument("c", "Note C", "[[Note A]] and [[Note D]]"),
            NoteDocument("d", "Note D", "[[Note C]] and [[Note E]]"),
            NoteDocument("e", "Note E", "[[Note D]]"),
        ]
        for n in notes:
            self.pkg.add_document(n)
        self.pkg.build_graph()

        metrics = self.pkg.compute_centrality_metrics()
        c_betweenness = metrics["c"]["betweenness"]
        d_betweenness = metrics["d"]["betweenness"]

        # C and D are bridges between clusters, should have significantly higher betweenness than leaf nodes
        self.assertGreater(c_betweenness, metrics["b"]["betweenness"])
        self.assertGreater(d_betweenness, metrics["e"]["betweenness"])

    def test_backlinks_query(self):
        doc_x = NoteDocument("doc-x", "Documentation X", "Links to [[API Reference]].")
        doc_y = NoteDocument("doc-y", "Tutorial Y", "Links to [[API Reference]].")
        doc_target = NoteDocument("api-ref", "API Reference", "Root API docs.")

        self.pkg.add_document(doc_x)
        self.pkg.add_document(doc_y)
        self.pkg.add_document(doc_target)
        self.pkg.build_graph()

        # Query backlinks by exact title
        backlinks = self.pkg.get_backlinks("API Reference")
        self.assertCountEqual(backlinks, ["doc-x", "doc-y"])

        # Query backlinks by slug
        backlinks_slug = self.pkg.get_backlinks("api-ref")
        self.assertCountEqual(backlinks_slug, ["doc-x", "doc-y"])

    def test_shortest_path_navigation(self):
        notes = [
            NoteDocument("start", "Start Note", "Points to [[Step One]]."),
            NoteDocument("step-1", "Step One", "Points to [[Step Two]]."),
            NoteDocument("step-2", "Step Two", "Points to [[Goal Note]]."),
            NoteDocument("goal", "Goal Note", "Destination reached."),
            NoteDocument("isolated", "Isolated Note", "No incoming or outgoing links."),
        ]
        for n in notes:
            self.pkg.add_document(n)
        self.pkg.build_graph()

        # Connected path
        path = self.pkg.find_shortest_path("start", "goal")
        self.assertEqual(path, ["start", "step-1", "step-2", "goal"])

        # Disconnected path
        no_path = self.pkg.find_shortest_path("start", "isolated")
        self.assertIsNone(no_path)

    def test_community_detection_greedy_modularity(self):
        notes = [
            # Community 1: Python stack
            NoteDocument("py-base", "Python Base", "[[Python OOP]] and [[Python Async]]"),
            NoteDocument("py-oop", "Python OOP", "[[Python Base]]"),
            NoteDocument("py-async", "Python Async", "[[Python Base]]"),
            # Community 2: Web frontend stack
            NoteDocument("web-html", "Web HTML", "[[Web CSS]] and [[Web JS]]"),
            NoteDocument("web-css", "Web CSS", "[[Web HTML]]"),
            NoteDocument("web-js", "Web JS", "[[Web HTML]]"),
        ]
        for n in notes:
            self.pkg.add_document(n)
        self.pkg.build_graph()

        communities = self.pkg.detect_communities()
        self.assertEqual(len(communities), 2)
        py_cluster = {"py-base", "py-oop", "py-async"}
        web_cluster = {"web-html", "web-css", "web-js"}
        self.assertIn(py_cluster, communities)
        self.assertIn(web_cluster, communities)


if __name__ == "__main__":
    unittest.main()
