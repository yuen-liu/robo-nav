"""
Topological Visual Graph Engine for LingBot-MAP
-----------------------------------------------
Provides a modular topological graph representation of visual scenes where:
- Nodes represent room/zones (e.g. office, printer, microkitchen)
- Edges represent hallways/transitions (e.g. hallway1, hallway2)

Features:
- Frame-to-node mapping (`get_node_from_frame`)
- BFS shortest path planning between nodes (`plan_path`)
- Waypoint keyframe extraction along planned routes (`get_path_waypoints`)
- Formatted navigation plan printing (`print_navigation_plan`)
"""

import os
from typing import Dict, List, Optional, Tuple, Set


class TopologicalGraph:
    """Topological visual graph representing room/zone nodes and hallway transition edges."""

    def __init__(self):
        # Default nodes (rooms/zones) and edges (hallways/transitions)
        self.nodes: Set[str] = {"office", "printer", "microkitchen"}
        self.hallways: Set[str] = {"hallway1", "hallway2"}

        # Graph adjacency mapping: node -> list of (neighbor_node, transition_edge, weight)
        # Spatial topology: office <-> hallway1 <-> printer <-> hallway2 <-> microkitchen
        self.adj: Dict[str, List[Tuple[str, str, float]]] = {
            "office": [("printer", "hallway1", 1.0)],
            "printer": [("office", "hallway1", 1.0), ("microkitchen", "hallway2", 1.0)],
            "microkitchen": [("printer", "hallway2", 1.0)],
        }

    def add_node(self, node_name: str) -> None:
        """Add a room/zone node to the graph."""
        node_clean = node_name.lower()
        self.nodes.add(node_clean)
        if node_clean not in self.adj:
            self.adj[node_clean] = []

    def add_edge(self, u: str, v: str, hallway_name: str, weight: float = 1.0) -> None:
        """Add an undirected transition edge (hallway) between two nodes."""
        u_clean, v_clean = u.lower(), v.lower()
        h_clean = hallway_name.lower()
        self.add_node(u_clean)
        self.add_node(v_clean)
        self.hallways.add(h_clean)
        self.adj[u_clean].append((v_clean, h_clean, weight))
        self.adj[v_clean].append((u_clean, h_clean, weight))

    def get_node_from_frame(self, frame_path: str) -> str:
        """Identify which node/zone or hallway transition a database keyframe path belongs to based on directory structure."""
        if not frame_path:
            return "unknown"
        norm_path = os.path.normpath(frame_path)
        parts = norm_path.split(os.sep)

        # 1. Search directory components backwards (excluding filename)
        for part in reversed(parts[:-1]):
            part_clean = part.lower()
            if part_clean in self.nodes or part_clean in self.hallways:
                return part_clean

        # 2. If frame is in an unmapped subfolder (e.g. 80_frames), cross-reference parent directory for zone subfolders
        filename = os.path.basename(norm_path)
        parent_dir = os.path.dirname(os.path.dirname(norm_path))
        if os.path.isdir(parent_dir):
            for root, _, files in os.walk(parent_dir):
                if filename in files and os.path.normpath(root) != os.path.dirname(norm_path):
                    for p in os.path.normpath(root).split(os.sep):
                        p_clean = p.lower()
                        if p_clean in self.nodes or p_clean in self.hallways:
                            return p_clean

        # Fallback to parent directory name
        return parts[-2].lower() if len(parts) >= 2 else "unknown"

    def _resolve_to_node(self, zone_name: str) -> str:
        """Resolve a hallway/transition name or path string to the nearest primary room node."""
        if not zone_name:
            return "unknown"

        # Check basename first (e.g., /path/to/microkitchen -> microkitchen)
        zone_clean = os.path.basename(os.path.normpath(zone_name)).lower()
        if zone_clean in self.nodes:
            return zone_clean

        # Search path components if a full path was passed
        parts = os.path.normpath(zone_name).split(os.sep)
        for part in reversed(parts):
            part_clean = part.lower()
            if part_clean in self.nodes:
                return part_clean
            if part_clean in self.hallways:
                zone_clean = part_clean
                break

        if zone_clean in self.nodes:
            return zone_clean
        if zone_clean == "hallway1":
            return "office"
        if zone_clean == "hallway2":
            return "printer"
        return zone_clean

    def plan_path(self, start_node: str, goal_node: str) -> Optional[List[Tuple[str, Optional[str]]]]:
        """Compute topological shortest path from start_node to goal_node.

        Returns:
            List of tuples: [(node_1, transition_to_next), (node_2, transition_to_next), ..., (goal_node, None)]
            or None if no valid path exists.
        """
        start_clean = self._resolve_to_node(start_node)
        goal_clean = self._resolve_to_node(goal_node)

        if start_clean == goal_clean:
            return [(start_clean, None)]

        # BFS shortest path search
        queue = [[(start_clean, None)]]
        visited = {start_clean}

        while queue:
            path = queue.pop(0)
            current_node, _ = path[-1]

            if current_node == goal_clean:
                return path

            for neighbor, edge_name, _ in self.adj.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path[-1] = (current_node, edge_name)
                    new_path.append((neighbor, None))
                    queue.append(new_path)

        return None

    def get_path_waypoints(
        self,
        topological_path: List[Tuple[str, Optional[str]]],
        map_candidate_paths: List[str]
    ) -> List[Dict[str, str]]:
        """Extract ordered keyframe waypoints along the planned topological route."""
        waypoints = []
        if not topological_path:
            return waypoints

        for step_idx, (node, edge) in enumerate(topological_path, 1):
            # Find representative frames matching this node or edge zone
            matching_frames = [
                p for p in map_candidate_paths
                if self.get_node_from_frame(p) == node or (edge and self.get_node_from_frame(p) == edge)
            ]
            if not matching_frames:
                # Retry matching just node
                matching_frames = [
                    p for p in map_candidate_paths
                    if self.get_node_from_frame(p) == node
                ]

            rep_frame = matching_frames[len(matching_frames) // 2] if matching_frames else "N/A"

            waypoints.append({
                "step": step_idx,
                "node": node,
                "transition": edge if edge else "ARRIVED (Goal Destination)",
                "keyframe": os.path.basename(rep_frame) if rep_frame != "N/A" else "N/A",
                "keyframe_full_path": rep_frame,
            })

        return waypoints

    def print_navigation_plan(
        self,
        start_node: str,
        goal_node: str,
        waypoints: List[Dict[str, str]]
    ) -> None:
        """Print clean, formatted navigation plan."""
        print(f"\n=======================================================")
        print(f" 🗺️  TOPOLOGICAL NAVIGATION PLAN")
        print(f" Start Location: {start_node.upper()}")
        print(f" Destination:    {goal_node.upper()}")
        print(f"=======================================================")
        if not waypoints:
            print(f" ⚠️ No topological route found between [{start_node.upper()}] and [{goal_node.upper()}].")
            print(f"   Registered graph nodes: {sorted(list(self.nodes))}")
            print(f"   Registered transition edges: {sorted(list(self.hallways))}")
            print(f"   Tip: Specify --map_folder ~/project/all_frames (containing zone subfolders office, printer, microkitchen, hallway1, hallway2).")
            print(f"=======================================================\n")
            return

        for wp in waypoints:
            step = wp["step"]
            node = wp["node"].upper()
            trans = wp["transition"]
            kf = wp["keyframe"]
            if trans == "ARRIVED (Goal Destination)":
                print(f" Step {step}: Reach [{node}] -> {trans} (Keyframe: {kf})")
            else:
                print(f" Step {step}: In [{node}] -> Transition via corridor [{trans}] (Keyframe: {kf})")
        print(f"=======================================================\n")
