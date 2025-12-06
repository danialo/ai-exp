#!/usr/bin/env python3
"""
TASK 6.3: Review Tentative Link Clusters

Build connected components of pending TentativeLinks.
Print clusters sorted by avg confidence and size.
This satisfies transitivity awareness without auto-merging.
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from sqlmodel import Session, select
from src.db import get_engine
from src.memory.models.belief_node import BeliefNode
from src.memory.models.tentative_link import TentativeLink


@dataclass
class Cluster:
    """A connected component of tentatively linked beliefs."""
    belief_ids: Set[str]
    links: List[TentativeLink]

    @property
    def size(self) -> int:
        return len(self.belief_ids)

    @property
    def avg_confidence(self) -> float:
        if not self.links:
            return 0.0
        return sum(link.confidence for link in self.links) / len(self.links)

    @property
    def min_confidence(self) -> float:
        if not self.links:
            return 0.0
        return min(link.confidence for link in self.links)

    @property
    def max_confidence(self) -> float:
        if not self.links:
            return 0.0
        return max(link.confidence for link in self.links)


def build_adjacency(links: List[TentativeLink]) -> Dict[str, Set[str]]:
    """Build adjacency list from tentative links."""
    adj: Dict[str, Set[str]] = defaultdict(set)
    for link in links:
        adj[link.from_belief_id].add(link.to_belief_id)
        adj[link.to_belief_id].add(link.from_belief_id)
    return adj


def find_connected_components(
    adj: Dict[str, Set[str]],
    links: List[TentativeLink]
) -> List[Cluster]:
    """Find connected components using BFS."""
    visited: Set[str] = set()
    clusters: List[Cluster] = []

    # Build link lookup for cluster assignment
    link_lookup: Dict[Tuple[str, str], TentativeLink] = {}
    for link in links:
        # Store both directions for lookup
        link_lookup[(link.from_belief_id, link.to_belief_id)] = link
        link_lookup[(link.to_belief_id, link.from_belief_id)] = link

    all_nodes = set(adj.keys())

    for start_node in all_nodes:
        if start_node in visited:
            continue

        # BFS from this node
        component_nodes: Set[str] = set()
        queue = [start_node]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue

            visited.add(node)
            component_nodes.add(node)

            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        # Collect links within this component
        component_links: List[TentativeLink] = []
        seen_link_ids: Set[str] = set()

        for node in component_nodes:
            for neighbor in adj[node]:
                if neighbor in component_nodes:
                    link = link_lookup.get((node, neighbor))
                    if link and link.link_id not in seen_link_ids:
                        component_links.append(link)
                        seen_link_ids.add(link.link_id)

        if component_nodes:
            clusters.append(Cluster(
                belief_ids=component_nodes,
                links=component_links
            ))

    return clusters


def get_belief_text(session: Session, belief_id: str) -> str:
    """Get canonical text for a belief."""
    stmt = select(BeliefNode).where(BeliefNode.belief_id == belief_id)
    belief = session.exec(stmt).first()
    return belief.canonical_text if belief else "<unknown>"


def print_cluster(
    cluster: Cluster,
    session: Session,
    index: int,
    verbose: bool = False
) -> None:
    """Print a cluster summary."""
    print(f"\n{'='*60}")
    print(f"CLUSTER {index}")
    print(f"{'='*60}")
    print(f"Size: {cluster.size} beliefs")
    print(f"Links: {len(cluster.links)}")
    print(f"Avg confidence: {cluster.avg_confidence:.3f}")
    print(f"Min confidence: {cluster.min_confidence:.3f}")
    print(f"Max confidence: {cluster.max_confidence:.3f}")

    print(f"\nBeliefs:")
    for i, belief_id in enumerate(sorted(cluster.belief_ids), 1):
        text = get_belief_text(session, belief_id)
        # Truncate long text
        if len(text) > 80:
            text = text[:77] + "..."
        print(f"  {i}. [{belief_id[:8]}] {text}")

    if verbose:
        print(f"\nLinks:")
        for link in sorted(cluster.links, key=lambda x: x.confidence, reverse=True):
            from_text = get_belief_text(session, link.from_belief_id)[:40]
            to_text = get_belief_text(session, link.to_belief_id)[:40]
            print(f"  [{link.confidence:.3f}] {from_text}...")
            print(f"           -> {to_text}...")
            print(f"           support_both={link.support_both} support_one={link.support_one}")


def main():
    parser = argparse.ArgumentParser(
        description="Review tentative link clusters for manual resolution"
    )
    parser.add_argument(
        "--status",
        default="pending",
        choices=["pending", "accepted", "rejected", "all"],
        help="Filter links by status (default: pending)"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=1,
        help="Minimum cluster size to display (default: 1)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum avg confidence to display (default: 0.0)"
    )
    parser.add_argument(
        "--sort-by",
        default="confidence",
        choices=["confidence", "size"],
        help="Sort clusters by (default: confidence)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed link information"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of clusters to display"
    )

    args = parser.parse_args()

    engine = get_engine()

    with Session(engine) as session:
        # Query tentative links
        stmt = select(TentativeLink)
        if args.status != "all":
            stmt = stmt.where(TentativeLink.status == args.status)

        links = list(session.exec(stmt).all())

        if not links:
            print(f"No tentative links found with status='{args.status}'")
            return

        print(f"Found {len(links)} tentative links with status='{args.status}'")

        # Build adjacency and find components
        adj = build_adjacency(links)
        clusters = find_connected_components(adj, links)

        # Filter by size and confidence
        clusters = [
            c for c in clusters
            if c.size >= args.min_size and c.avg_confidence >= args.min_confidence
        ]

        # Sort
        if args.sort_by == "confidence":
            clusters.sort(key=lambda x: (x.avg_confidence, x.size), reverse=True)
        else:
            clusters.sort(key=lambda x: (x.size, x.avg_confidence), reverse=True)

        # Apply limit
        if args.limit:
            clusters = clusters[:args.limit]

        # Summary
        print(f"\nFound {len(clusters)} clusters meeting criteria")
        print(f"Total beliefs involved: {sum(c.size for c in clusters)}")

        # Print clusters
        for i, cluster in enumerate(clusters, 1):
            print_cluster(cluster, session, i, verbose=args.verbose)

        # Summary stats
        if clusters:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            print(f"Total clusters: {len(clusters)}")
            print(f"Largest cluster: {max(c.size for c in clusters)} beliefs")
            print(f"Highest avg confidence: {max(c.avg_confidence for c in clusters):.3f}")
            print(f"Lowest avg confidence: {min(c.avg_confidence for c in clusters):.3f}")


if __name__ == "__main__":
    main()
