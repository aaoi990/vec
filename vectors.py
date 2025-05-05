from datasketch import MinHash, MinHashLSHForest

samples = {
    "server_1": {
        "headers": {
            "Server": "nginx",
            "Content-Type": "text/html",
            "X-Frame-Options": "SAMEORIGIN"
        },
        "tls": {
            "tls_version": "1.3",
            "cipher_suite": "TLS_AES_256_GCM_SHA384"
        }
    },
    "server_2": {
        "headers": {
            "Server": "apache",
            "Content-Type": "text/html",
            "X-Content-Type-Options": "nosniff"
        },
        "tls": {
            "tls_version": "1.2",
            "cipher_suite": "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
        }
    },
    "server_3": {
        "headers": {
            "Server": "nginx",
            "Content-Type": "text/html",
            "X-Frame-Options": "SAMEORIGIN"
        },
        "tls": {
            "tls_version": "1.3",
            "cipher_suite": "TLS_AES_256_GCM_SHA384"
        }
    }
}


# ---------- 2. Feature extraction function ----------
def extract_features(headers: dict, tls: dict) -> set:
    features = set()
    for k, v in headers.items():
        features.add(f"header:{k}:{v}")
    for k, v in tls.items():
        features.add(f"tls:{k.lower()}:{v.lower()}")
    
    print(features)
    return features


# ---------- 3. Create MinHash signatures ----------
def create_minhash(features: set, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for feature in features:
        m.update(feature.encode("utf8"))
    return m

# Store MinHashes for all samples
minhashes = {}
for server_id, data in samples.items():
    features = extract_features(data["headers"], data["tls"])
    minhashes[server_id] = create_minhash(features)


# ---------- 4. Build LSH Forest ----------
forest = MinHashLSHForest(num_perm=128)
for server_id, m in minhashes.items():
    forest.add(server_id, m)
forest.index()


# ---------- 5. Query for similar servers ----------
query_id = "server_1"
result = forest.query(minhashes[query_id], 3)
print(result)

print(f"Top similar servers to {query_id}:")
for match in result:
    if match != query_id:
        print(f" - {match}")


for match in result:
    if match != query_id:
        # Directly retrieve and print the Jaccard similarity score between the query server and each match
        jaccard_sim = minhashes[query_id].jaccard(minhashes[match])
        print(f" - {match}: Jaccard similarity = {jaccard_sim:.4f}")