from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class FabricConfig:
    channel_name: str = "health-data-channel"
    chaincode_name: str = "dataset_contract"
    # In a full implementation, you would also have:
    # - connection profile path
    # - msp ID
    # - crypto material paths


class FabricClient:
    """Thin wrapper around Hyperledger Fabric SDK.

    In this scaffold we only define interfaces; actual network interaction
    should be implemented using the Fabric Python/Node SDK or REST gateway.
    """

    def __init__(self, config: FabricConfig | None = None) -> None:
        self.config = config or FabricConfig()

    # ---- Dataset governance ----

    def register_dataset(self, payload: Dict[str, Any]) -> str:
        """Invoke registerDataset on chaincode."""
        # TODO: integrate with Fabric Gateway / SDK
        tx_id = f"mock-tx-register-{payload.get('dataset_id')}"
        return tx_id

    def request_dataset_access(self, payload: Dict[str, Any]) -> str:
        tx_id = f"mock-tx-request-{payload.get('request_id')}"
        return tx_id

    def approve_access(self, payload: Dict[str, Any]) -> str:
        tx_id = f"mock-tx-approve-{payload.get('request_id')}"
        return tx_id

    def reject_access(self, payload: Dict[str, Any]) -> str:
        tx_id = f"mock-tx-reject-{payload.get('request_id')}"
        return tx_id

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        return {"dataset_id": dataset_id, "status": "mock"}

    def get_access_requests(self, dataset_id: str) -> List[Dict[str, Any]]:
        return []

