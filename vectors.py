from datasketch import MinHash, MinHashLSHForest
from typing import Dict, Set, List, Tuple, Any
import json
from collections import defaultdict

class ServerSimilarityAnalyzer:
    """
    A class to analyze similarity between servers based on their headers and TLS configurations.
    Optimized for comparing thousands of servers efficiently.
    """
    
    def __init__(self, num_perm=128):
        """
        Initialize the analyzer with the specified number of permutations for MinHash.
        
        Args:
            num_perm: Number of permutations for MinHash. Higher values give more accurate
                     similarity estimates but use more memory.
        """
        self.num_perm = num_perm
        self.servers = {}
        self.header_minhashes = {}
        self.tls_minhashes = {}
        self.combined_minhashes = {}
        
        # Separate forests for each component
        self.header_forest = MinHashLSHForest(num_perm=num_perm)
        self.tls_forest = MinHashLSHForest(num_perm=num_perm)
        self.combined_forest = MinHashLSHForest(num_perm=num_perm)
        
        self.indexed = False
    
    def add_server(self, server_id: str, headers: dict, tls: dict) -> None:
        """
        Add a server to the analyzer.
        
        Args:
            server_id: Unique identifier for the server
            headers: Dictionary of HTTP headers
            tls: Dictionary of TLS configuration parameters
        """
        self.servers[server_id] = {"headers": headers, "tls": tls}
        
        # Extract features and create MinHashes
        header_features = self._extract_component_features(headers, "header")
        tls_features = self._extract_component_features(tls, "tls")
        combined_features = header_features.union(tls_features)
        
        # Create MinHashes
        self.header_minhashes[server_id] = self._create_minhash(header_features)
        self.tls_minhashes[server_id] = self._create_minhash(tls_features)
        self.combined_minhashes[server_id] = self._create_minhash(combined_features)
        
        # Add to forests (but don't index yet)
        self.header_forest.add(server_id, self.header_minhashes[server_id])
        self.tls_forest.add(server_id, self.tls_minhashes[server_id])
        self.combined_forest.add(server_id, self.combined_minhashes[server_id])
        
        # Mark as not indexed
        self.indexed = False
    
    def add_servers_batch(self, servers_data: Dict[str, Dict[str, Dict]]) -> None:
        """
        Add multiple servers at once.
        
        Args:
            servers_data: Dictionary mapping server_id to a dict with 'headers' and 'tls' keys
        """
        for server_id, data in servers_data.items():
            self.add_server(server_id, data.get("headers", {}), data.get("tls", {}))
    
    def build_index(self) -> None:
        """
        Build the LSH Forest index. Must be called after adding servers and before querying.
        """
        self.header_forest.index()
        self.tls_forest.index()
        self.combined_forest.index()
        self.indexed = True
    
    def find_similar_servers(self, query_id: str, num_results: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Find servers similar to the specified query server.
        
        Args:
            query_id: ID of the server to use as the query
            num_results: Maximum number of results to return
            
        Returns:
            Dictionary with similarity scores for headers, TLS, and combined
        """
        if not self.indexed:
            self.build_index()
        
        if query_id not in self.servers:
            raise ValueError(f"Server {query_id} not found")
        
        # Query each forest
        header_matches = self.header_forest.query(self.header_minhashes[query_id], num_results + 1)
        tls_matches = self.tls_forest.query(self.tls_minhashes[query_id], num_results + 1)
        combined_matches = self.combined_forest.query(self.combined_minhashes[query_id], num_results + 1)
        
        # Calculate similarity scores
        result = {}
        
        # Process header matches
        for match in header_matches:
            if match != query_id:
                if match not in result:
                    result[match] = {"header": 0.0, "tls": 0.0, "combined": 0.0}
                result[match]["header"] = self.header_minhashes[query_id].jaccard(self.header_minhashes[match])
        
        # Process TLS matches
        for match in tls_matches:
            if match != query_id:
                if match not in result:
                    result[match] = {"header": 0.0, "tls": 0.0, "combined": 0.0}
                result[match]["tls"] = self.tls_minhashes[query_id].jaccard(self.tls_minhashes[match])
        
        # Process combined matches
        for match in combined_matches:
            if match != query_id:
                if match not in result:
                    result[match] = {"header": 0.0, "tls": 0.0, "combined": 0.0}
                result[match]["combined"] = self.combined_minhashes[query_id].jaccard(self.combined_minhashes[match])
        
        # Sort by combined similarity and limit results
        sorted_results = {k: v for k, v in sorted(
            result.items(), 
            key=lambda item: item[1]["combined"], 
            reverse=True
        )[:num_results]}
        
        return sorted_results
    
    def calculate_pairwise_similarity(self, server1_id: str, server2_id: str) -> Dict[str, float]:
        """
        Calculate similarity metrics between two specific servers.
        
        Args:
            server1_id: ID of first server
            server2_id: ID of second server
            
        Returns:
            Dictionary with header_similarity, tls_similarity, and combined_similarity
        """
        if server1_id not in self.servers or server2_id not in self.servers:
            missing = []
            if server1_id not in self.servers:
                missing.append(server1_id)
            if server2_id not in self.servers:
                missing.append(server2_id)
            raise ValueError(f"Servers not found: {', '.join(missing)}")
        
        header_sim = self.header_minhashes[server1_id].jaccard(self.header_minhashes[server2_id])
        tls_sim = self.tls_minhashes[server1_id].jaccard(self.tls_minhashes[server2_id])
        combined_sim = self.combined_minhashes[server1_id].jaccard(self.combined_minhashes[server2_id])
        
        return {
            "header_similarity": header_sim,
            "tls_similarity": tls_sim,
            "combined_similarity": combined_sim
        }
    
    def calculate_all_pairwise_similarities(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Calculate pairwise similarities between all servers.
        Warning: This can be computationally expensive with many servers.
        
        Returns:
            Dictionary mapping pairs of server IDs to their similarity metrics
        """
        results = {}
        server_ids = list(self.servers.keys())
        
        for i, server1_id in enumerate(server_ids):
            for server2_id in server_ids[i+1:]:  # Only calculate each pair once
                pair = (server1_id, server2_id)
                results[pair] = self.calculate_pairwise_similarity(server1_id, server2_id)
        
        return results
    
    def find_clusters(self, similarity_threshold: float = 0.7) -> List[List[str]]:
        """
        Group servers into clusters based on similarity threshold.
        
        Args:
            similarity_threshold: Minimum combined similarity to consider servers as part of the same cluster
            
        Returns:
            List of clusters, where each cluster is a list of server IDs
        """
        # Calculate all pairwise similarities
        similarities = self.calculate_all_pairwise_similarities()
        
        # Create adjacency map based on threshold
        adjacency = defaultdict(set)
        for (server1, server2), scores in similarities.items():
            if scores["combined_similarity"] >= similarity_threshold:
                adjacency[server1].add(server2)
                adjacency[server2].add(server1)
        
        # Find clusters using DFS
        visited = set()
        clusters = []
        
        def dfs(node, cluster):
            visited.add(node)
            cluster.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)
        
        for server_id in self.servers:
            if server_id not in visited:
                cluster = []
                dfs(server_id, cluster)
                clusters.append(cluster)
        
        return clusters
    
    def save_to_file(self, filename: str) -> None:
        """
        Save server data to a file.
        Note: Only saves the raw server data, not the MinHashes or forests.
        """
        with open(filename, 'w') as f:
            json.dump(self.servers, f)
    
    def load_from_file(self, filename: str) -> None:
        """
        Load server data from a file and build the analyzer.
        """
        with open(filename, 'r') as f:
            servers_data = json.load(f)
        
        # Reset analyzer
        self.__init__(self.num_perm)
        
        # Add servers
        self.add_servers_batch(servers_data)
        
        # Build index
        self.build_index()
    
    def _extract_component_features(self, component_dict: dict, prefix: str) -> Set[str]:
        """Extract features from a component dictionary with proper prefix."""
        features = set()
        for k, v in component_dict.items():
            # Normalize keys and values
            key = k
            # Handle different value types
            if isinstance(v, (dict, list)):
                val = json.dumps(v, sort_keys=True)
            else:
                val = str(v).lower() if isinstance(v, str) else str(v)
            
            features.add(key)
            features.add(val)
        
        return features
    
    def _create_minhash(self, features: Set[str]) -> MinHash:
        """Create a MinHash from a set of features."""
        m = MinHash(num_perm=self.num_perm)
        for feature in features:
            m.update(feature.encode('utf8'))
        return m


# Example usage
if __name__ == "__main__":
    # Sample data
    samples = {
        "server_1": {
            "headers": {
                "HTTP/1.1": "302 Found",
               
            },
            "tls": {
                "version": "TLSv1.3",
               
            }
        },
        "server_2": {
            "headers": {
                "HTTP/1.1": "302 Found",
                
            },
            "tls": {
                "version": "TLSv1.2",
              
            }
        },
        "server_3": {
            "headers": {
                "Server": "Apache/2.4.41",
            },
            "tls": {
                "version": "TLSv1.2",
            }
        }
    }
    
    # Create analyzer and add servers
    analyzer = ServerSimilarityAnalyzer(num_perm=2048)
    analyzer.add_servers_batch(samples)
    
    # Build index
    analyzer.build_index()
    
    # Find similar servers
    similar_to_server1 = analyzer.find_similar_servers("server_2", num_results=2)
    print("Servers similar to server_2:")
    for server_id, scores in similar_to_server1.items():
        print(f"  {server_id}: headers={scores['header']:.4f}, tls={scores['tls']:.4f}, combined={scores['combined']:.4f}")
    
    # Calculate specific pairwise similarity
    similarity = analyzer.calculate_pairwise_similarity("server_1", "server_2")
    print("\nSimilarity between server_1 and server_2:")
    print(f"  Headers: {similarity['header_similarity']:.4f}")
    print(f"  TLS: {similarity['tls_similarity']:.4f}")
    print(f"  Combined: {similarity['combined_similarity']:.4f}")
    
    # Find clusters
    clusters = analyzer.find_clusters(similarity_threshold=0.5)
    print("\nServer clusters (similarity >= 0.5):")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {', '.join(cluster)}")