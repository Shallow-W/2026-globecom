"""Topology class for network topology with Dijkstra shortest path."""

import heapq
from typing import Dict, List, Tuple, Optional

from .node import Node
from .link import Link


class Topology:
    """网络拓扑"""

    def __init__(self):
        """Initialize an empty topology."""
        self.nodes: Dict[str, Node] = {}
        self.links: List[Link] = []
        self.adjacency: Dict[str, List[str]] = {}  # 邻接表
        self.link_map: Dict[Tuple[str, str], Link] = {}  # 快速查找链路

    def add_node(self, node: Node) -> None:
        """
        Add a node to the topology.

        Args:
            node: Node object to add
        """
        self.nodes[node.node_id] = node
        self.adjacency[node.node_id] = []

    def add_link(self, link: Link) -> None:
        """
        Add a link to the topology.

        Args:
            link: Link object to add
        """
        self.links.append(link)
        self.adjacency[link.src].append(link.dst)
        self.link_map[(link.src, link.dst)] = link

    def get_shortest_path(self, src: str, dst: str) -> List[str]:
        """
        Get shortest path between two nodes using Dijkstra's algorithm.

        Args:
            src: Source node ID
            dst: Destination node ID

        Returns:
            List of node IDs forming the path, empty if no path exists
        """
        if src not in self.nodes or dst not in self.nodes:
            return []

        distances = {n: float('inf') for n in self.nodes}
        distances[src] = 0
        previous = {n: None for n in self.nodes}
        pq = [(0, src)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > distances[u]:
                continue
            if u == dst:
                break
            for v in self.adjacency.get(u, []):
                link = self.link_map.get((u, v))
                if link:
                    nd = d + link.delay
                    if nd < distances[v]:
                        distances[v] = nd
                        previous[v] = u
                        heapq.heappush(pq, (nd, v))

        # 重建路径
        if previous[dst] is None and dst != src:
            return []
        path = []
        cur = dst
        while cur:
            path.append(cur)
            cur = previous[cur]
        path.reverse()
        return path

    def get_communication_delay(self, src: str, dst: str) -> float:
        """
        Get communication delay between two nodes.

        Args:
            src: Source node ID
            dst: Destination node ID

        Returns:
            Total delay in ms, 0 if same node or no path
        """
        if src == dst:
            return 0.0
        path = self.get_shortest_path(src, dst)
        if not path or len(path) < 2:
            return 0.0
        return sum(self.link_map[(path[i], path[i+1])].delay for i in range(len(path)-1))

    def get_nodes_by_resource(self, cpu_needed: int, gpu_needed: int = 0) -> List[Node]:
        """
        Get all nodes that can satisfy resource requirements.

        Args:
            cpu_needed: CPU cores required
            gpu_needed: GPU memory in MB required

        Returns:
            List of nodes with sufficient resources
        """
        return [n for n in self.nodes.values() if n.can_deploy(cpu_needed, gpu_needed)]

    def __repr__(self) -> str:
        return f"Topology(nodes={len(self.nodes)}, links={len(self.links)})"
