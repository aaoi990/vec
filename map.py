
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
import random
from typing import Dict, List, Tuple, Optional
import community as community_louvain

#uses kruskals and mst

class LSHVisualizer:
    """
    Class to visualize the LSH Forest as a minimum spanning tree (MST) similar to t-map.
    """
    
    def __init__(self, analyzer):
        """
        Initialize with a ServerSimilarityAnalyzer instance.
        
        Args:
            analyzer: An instance of ServerSimilarityAnalyzer
        """
        self.analyzer = analyzer
        self.graph = None
        self.positions = None
        self.communities = None
        
    def build_similarity_graph(self, threshold: float = 0.0) -> nx.Graph:
        """
        Build a graph where nodes are servers and edges represent similarity.
        
        Args:
            threshold: Minimum similarity to create an edge (0.0-1.0)
            
        Returns:
            A NetworkX graph of server relationships
        """
        # Ensure we have calculated all pairwise similarities
        similarities = self.analyzer.calculate_all_pairwise_similarities()
        
        # Create graph
        G = nx.Graph()
        
        # Add all servers as nodes
        for server_id in self.analyzer.servers:
            # Add node attributes like types of headers, etc.
            server_data = self.analyzer.servers[server_id]
            G.add_node(server_id, headers=len(server_data['headers']), 
                      tls=len(server_data['tls']))
        
        # Add edges for similar servers
        for (server1, server2), scores in similarities.items():
            combined_sim = scores['combined_similarity']
            if combined_sim >= threshold:
                # Use distance (1-similarity) as weight for MST calculation
                G.add_edge(server1, server2, 
                           weight=1.0-combined_sim,  # Lower weight = stronger connection
                           similarity=combined_sim,
                           header_sim=scores['header_similarity'],
                           tls_sim=scores['tls_similarity'])
        
        self.graph = G
        return G
    
    def create_mst(self) -> nx.Graph:
        """
        Create a minimum spanning tree from the similarity graph.
        
        Returns:
            A NetworkX graph representing the MST
        """
        if self.graph is None:
            self.build_similarity_graph()
            
        # Create MST using weights (which are 1-similarity)
        mst = nx.minimum_spanning_tree(self.graph, weight='weight')
        return mst
    
    def detect_communities(self, resolution: float = 1.0) -> Dict[str, int]:
        """
        Detect communities of servers using Louvain algorithm.
        
        Args:
            resolution: Resolution parameter for community detection (higher values give smaller communities)
            
        Returns:
            Dictionary mapping server IDs to community IDs
        """
        if self.graph is None:
            self.build_similarity_graph()
            
        # Apply Louvain community detection
        self.communities = community_louvain.best_partition(self.graph, weight='similarity', 
                                                          resolution=resolution)
        return self.communities
    
    def calculate_2d_layout(self, method: str = 'mds') -> Dict[str, Tuple[float, float]]:
        """
        Calculate 2D positions for nodes based on their similarity distances.
        
        Args:
            method: 'mds' for Multidimensional Scaling or 'spring' for force-directed
            
        Returns:
            Dictionary mapping server IDs to (x, y) positions
        """
        if self.graph is None:
            self.build_similarity_graph()
            
        if method == 'mds':
            # Create distance matrix from similarity
            n = len(self.graph.nodes)
            dist_matrix = np.ones((n, n)) * 0.5  # Default distance
            
            # Map node IDs to indices
            nodes = list(self.graph.nodes)
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            # Fill in actual distances where we know them
            for u, v, data in self.graph.edges(data=True):
                i, j = node_to_idx[u], node_to_idx[v]
                dist_matrix[i, j] = dist_matrix[j, i] = 1.0 - data['similarity']
                
            # Use MDS to compute 2D coordinates
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            positions = mds.fit_transform(dist_matrix)
            
            # Map back to node IDs
            self.positions = {nodes[i]: (positions[i, 0], positions[i, 1]) for i in range(n)}
            
        elif method == 'spring':
            # Use force-directed layout
            self.positions = nx.spring_layout(self.graph, weight='weight', seed=42)
            
        return self.positions
    
    def visualize(self, title: str = "Server Similarity MST", 
              show_mst: bool = True,
              community_colors: bool = True,
              edge_threshold: float = 0.0,
              node_size_factor: int = 100,
              show_edge_weights: bool = False,
              highlight_servers: Optional[List[str]] = None,
              filename: Optional[str] = None):
        """
        Create and save visualization of server relationships to a file.
        
        Args:
            title: Title for the visualization
            show_mst: If True, only show MST edges; if False, show all edges above threshold
            community_colors: If True, color nodes by detected communities
            edge_threshold: Minimum similarity to display edges (if not showing MST)
            node_size_factor: Base factor for node size
            show_edge_weights: If True, show similarity values on edges
            highlight_servers: List of server IDs to highlight
            filename: If provided, save visualization to this file
        """
        # Set the backend to Agg (a non-interactive backend) before importing pyplot
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if self.graph is None:
            self.build_similarity_graph(threshold=edge_threshold)
            
        if self.positions is None:
            self.calculate_2d_layout()
            
        # Determine which graph to visualize
        if show_mst:
            viz_graph = self.create_mst()
        else:
            # Filter graph to only include edges above threshold
            viz_graph = self.graph.copy()
            edges_to_remove = [(u, v) for u, v, d in viz_graph.edges(data=True) 
                            if d['similarity'] < edge_threshold]
            viz_graph.remove_edges_from(edges_to_remove)
        
        plt.figure(figsize=(12, 10))
        
        # Determine node colors based on communities if requested
        if community_colors and self.communities is None:
            self.detect_communities()
            
        if community_colors and self.communities is not None:
            # Map community IDs to colors
            unique_communities = set(self.communities.values())
            color_map = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
            community_to_color = {com: color_map[i] for i, com in enumerate(unique_communities)}
            
            # Create node color list
            node_colors = [community_to_color[self.communities[node]] for node in viz_graph.nodes()]
        else:
            # Default color
            node_colors = 'skyblue'
        
        # Determine node sizes based on feature count
        node_sizes = [node_size_factor * (self.graph.nodes[n]['headers'] + 
                                        self.graph.nodes[n]['tls']) 
                    for n in viz_graph.nodes()]
        
        # Highlight specific servers if requested
        if highlight_servers:
            node_borders = []
            border_widths = []
            for node in viz_graph.nodes():
                if node in highlight_servers:
                    node_borders.append('red')
                    border_widths.append(2.0)
                else:
                    node_borders.append('black')
                    border_widths.append(0.5)
        else:
            node_borders = 'black'
            border_widths = 0.5
            
        # Draw nodes
        nx.draw_networkx_nodes(viz_graph, self.positions, 
                            node_color=node_colors,
                            node_size=node_sizes,
                            edgecolors=node_borders,
                            linewidths=border_widths)
        
        # Draw edges with width proportional to similarity
        edge_weights = [d['similarity'] * 2 for u, v, d in viz_graph.edges(data=True)]
        nx.draw_networkx_edges(viz_graph, self.positions, width=edge_weights, 
                            alpha=0.7, edge_color='gray')
        
        # Add labels
        nx.draw_networkx_labels(viz_graph, self.positions, font_size=8)
        
        # Show edge weights if requested
        if show_edge_weights:
            edge_labels = {(u, v): f"{d['similarity']:.2f}" for u, v, d in viz_graph.edges(data=True)}
            nx.draw_networkx_edge_labels(viz_graph, self.positions, edge_labels=edge_labels, font_size=6)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        # If no filename was provided, generate one
        if not filename:
            import time
            timestamp = int(time.time())
            filename = f"server_graph_{timestamp}.png"
            
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Graph saved to: {filename}")
        
        # Close the figure to free memory
        plt.close()

    def generate_similarity_report(self, threshold: float = 0.5) -> Dict:
        """
        Generate a report of server clusters and their relationships.
        
        Args:
            threshold: Minimum similarity to include in the report
            
        Returns:
            Dictionary containing cluster information and statistics
        """
        if self.communities is None:
            self.detect_communities()
            
        # Organize servers by community
        community_groups = {}
        for server, community_id in self.communities.items():
            if community_id not in community_groups:
                community_groups[community_id] = []
            community_groups[community_id].append(server)
        
        # Analyze intra-community similarities
        community_stats = {}
        for community_id, servers in community_groups.items():
            similarities = []
            for i, server1 in enumerate(servers):
                for server2 in servers[i+1:]:
                    sim = self.analyzer.calculate_pairwise_similarity(server1, server2)['combined_similarity']
                    similarities.append(sim)
            
            if similarities:
                community_stats[community_id] = {
                    'servers': servers,
                    'size': len(servers),
                    'avg_similarity': sum(similarities) / len(similarities) if similarities else 0,
                    'min_similarity': min(similarities) if similarities else 0,
                    'max_similarity': max(similarities) if similarities else 0
                }
        
        # Find relationships between communities
        community_connections = {}
        for (server1, server2), scores in self.analyzer.calculate_all_pairwise_similarities().items():
            if scores['combined_similarity'] >= threshold:
                comm1 = self.communities[server1]
                comm2 = self.communities[server2]
                if comm1 != comm2:
                    key = tuple(sorted([comm1, comm2]))
                    if key not in community_connections:
                        community_connections[key] = []
                    community_connections[key].append((server1, server2, scores['combined_similarity']))
        
        # Prepare report
        report = {
            'num_servers': len(self.analyzer.servers),
            'num_communities': len(community_groups),
            'community_stats': community_stats,
            'inter_community_connections': community_connections
        }
        
        return report
    
# Example usage
if __name__ == "__main__":
    from vectors import ServerSimilarityAnalyzer
    
    # Sample data - many more servers in a real scenario
    samples = {
        
        # Add more servers as needed
    }
    
    # Create analyzer and add servers
    analyzer = ServerSimilarityAnalyzer(num_perm=128)
    analyzer.add_servers_batch(samples)
    analyzer.build_index()
    
    # Create visualizer
    visualizer = LSHVisualizer(analyzer)
    
    # Create and show visualization
    visualizer.build_similarity_graph(threshold=0.1)
    visualizer.detect_communities()
    visualizer.calculate_2d_layout(method='mds')
    
    visualizer.visualize(
    title="Malware Server Infrastructure Map", 
    filename="malware_mst.png",
    show_mst=True,
    community_colors=True,
    highlight_servers=["malware_server_1"]
)
    # Show full graph with threshold
    visualizer.visualize(
        title="Server Similarity Network (connections > 0.4)", 
        filename="server_network.png",
        show_mst=False,
        edge_threshold=0.4,
        community_colors=True
        
    )
    
    # Generate and print report
    report = visualizer.generate_similarity_report(threshold=0.3)
    print(f"Found {report['num_communities']} server communities")
    
    for comm_id, stats in report['community_stats'].items():
        print(f"Community {comm_id}: {len(stats['servers'])} servers, avg similarity: {stats['avg_similarity']:.2f}")
        print(f"  Servers: {', '.join(stats['servers'])}")