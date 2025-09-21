from cookbook.constants import BEAKER_KNOWN_CLUSTERS, NEW_CLUSTER_ALIASES


def get_matching_clusters(cluster: str) -> list[str]:
    """
    This function converts cluster aliases to the actual cluster names; it also
    handles the cases where a cluster is referred to by an alias.
    """
    if cluster in NEW_CLUSTER_ALIASES:
        cluster = NEW_CLUSTER_ALIASES[cluster]

    if cluster in BEAKER_KNOWN_CLUSTERS:
        return BEAKER_KNOWN_CLUSTERS[cluster]

    return [cluster]


def is_gcs_cluster(cluster: str) -> bool:
    """
    This function checks if a cluster has GCS support; support here means we don't have to
    push google credentials to access the cluster.
    """

    canonical_names = get_matching_clusters(cluster)

    if all(cluster_name in BEAKER_KNOWN_CLUSTERS["goog"] for cluster_name in canonical_names):
        return True

    return False


def get_known_clusters() -> list[str]:
    """
    This function returns all known clusters in OLMo Cookbook.
    """
    all_clusters = [c for cs in BEAKER_KNOWN_CLUSTERS.values() for c in cs]
    return sorted(set(all_clusters))
