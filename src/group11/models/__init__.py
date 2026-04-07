from .gcn import GCNLatencyPredictor
from .gat import GATLatencyPredictor
from .mpnn import MPNNLatencyPredictor
from .routenet_fermi import RouteNetFermiPredictor
from .edge_gnn import EdgeWeightGNN
from .temporal_gnn import SpatialTemporalGNN

__all__ = [
    "GCNLatencyPredictor",
    "GATLatencyPredictor",
    "MPNNLatencyPredictor",
    "RouteNetFermiPredictor",
    "EdgeWeightGNN",
    "SpatialTemporalGNN",
]
