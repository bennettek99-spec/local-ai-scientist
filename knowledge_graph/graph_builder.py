"""Knowledge graph linking papers, authors, topics, and concepts.

The graph is rebuilt on demand from the paper database. Nodes are typed
(``paper``, ``author``, ``topic``, ``field``) and edges express authorship,
field membership, and topical relatedness. An interactive HTML visualisation is
produced with pyvis.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from config.settings import settings
from database.paper_database import PaperDatabase
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Colours used by the pyvis visualisation, keyed by node type.
_NODE_COLORS = {
    "paper": "#4C9BE8",
    "author": "#E8804C",
    "topic": "#5FB97A",
    "field": "#B05FE8",
}


class KnowledgeGraphBuilder:
    """Build, persist, and visualise the research knowledge graph."""

    def __init__(self, database: PaperDatabase | None = None) -> None:
        self.db = database or PaperDatabase()
        self.graph = nx.Graph()

    # ------------------------------------------------------------------- build
    def build(self, field: str | None = None, limit: int | None = None) -> nx.Graph:
        """(Re)build the graph from stored papers and their analyses."""
        self.graph = nx.Graph()
        papers = self.db.list_papers(field=field, limit=limit)
        logger.info("Building knowledge graph from %d papers", len(papers))

        for paper in papers:
            paper_node = f"paper::{paper.arxiv_id}"
            self.graph.add_node(
                paper_node,
                label=_truncate(paper.title, 60),
                type="paper",
                arxiv_id=paper.arxiv_id,
                field=paper.field,
                title=paper.title,
            )

            # Field membership.
            if paper.field:
                field_node = f"field::{paper.field}"
                self._ensure_node(field_node, paper.field, "field")
                self.graph.add_edge(paper_node, field_node, relation="in_field")

            # Authorship.
            for author in paper.authors:
                author_node = f"author::{author}"
                self._ensure_node(author_node, author, "author")
                self.graph.add_edge(paper_node, author_node, relation="authored_by")

            # Topics / concepts come from the analysis (or fall back to categories).
            analysis = self.db.get_analysis(paper.arxiv_id)
            topics = (analysis.related_topics if analysis else []) or paper.categories
            for topic in topics:
                topic_clean = topic.strip().lower()
                if not topic_clean:
                    continue
                topic_node = f"topic::{topic_clean}"
                self._ensure_node(topic_node, topic.strip(), "topic")
                self.graph.add_edge(paper_node, topic_node, relation="about")

        logger.info(
            "Graph built: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    def _ensure_node(self, node_id: str, label: str, node_type: str) -> None:
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, label=label, type=node_type)

    # -------------------------------------------------------------- statistics
    def stats(self) -> dict[str, int]:
        """Return counts of nodes by type plus total edges."""
        counts: dict[str, int] = {"edges": self.graph.number_of_edges()}
        for _, data in self.graph.nodes(data=True):
            key = data.get("type", "unknown")
            counts[key] = counts.get(key, 0) + 1
        return counts

    def central_topics(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Return the most-connected topic nodes as (label, degree) tuples."""
        topics = [
            (data.get("label", node), self.graph.degree(node))
            for node, data in self.graph.nodes(data=True)
            if data.get("type") == "topic"
        ]
        topics.sort(key=lambda item: item[1], reverse=True)
        return topics[:top_n]

    def related_papers(self, arxiv_id: str) -> list[str]:
        """Find papers connected through shared topics or authors."""
        paper_node = f"paper::{arxiv_id}"
        if not self.graph.has_node(paper_node):
            return []
        related: set[str] = set()
        for neighbour in self.graph.neighbors(paper_node):
            for second in self.graph.neighbors(neighbour):
                data = self.graph.nodes[second]
                if data.get("type") == "paper" and second != paper_node:
                    related.add(data.get("arxiv_id", ""))
        return sorted(r for r in related if r)

    # ----------------------------------------------------------------- persist
    def save_graphml(self, path: Path | None = None) -> Path:
        """Persist the graph to GraphML for reuse in other tools."""
        path = path or settings.graph_path
        nx.write_graphml(self.graph, path)
        logger.info("Saved knowledge graph to %s", path)
        return path

    def to_pyvis_html(self, path: Path | None = None) -> Path | None:
        """Render an interactive HTML visualisation with pyvis.

        Returns the output path, or None if pyvis is not installed.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            logger.warning("pyvis not installed; skipping HTML visualisation.")
            return None

        path = path or (settings.data_dir / "knowledge_graph.html")
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#222222",
            notebook=False,
        )
        net.barnes_hut()
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            net.add_node(
                node,
                label=data.get("label", node),
                color=_NODE_COLORS.get(node_type, "#999999"),
                title=f"{node_type}: {data.get('label', node)}",
                shape="dot" if node_type != "paper" else "square",
            )
        for source, target, data in self.graph.edges(data=True):
            net.add_edge(source, target, title=data.get("relation", ""))

        net.write_html(str(path), open_browser=False, notebook=False)
        logger.info("Saved interactive graph to %s", path)
        return path
