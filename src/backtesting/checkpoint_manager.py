"""
Checkpoint Manager

Save and load backtest state for resuming and branching.
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class CheckpointManager:
    """
    Manage checkpoint save/load operations

    Checkpoints preserve:
    - Account state (cash, positions, metrics)
    - Trade history
    - LLM response cache (optional)
    - Metadata (model, dates, iterations)
    """

    def __init__(self, checkpoint_dir: str = "results/checkpoints"):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        # Don't create the directory here - let save_checkpoint create nested structure

        logger.info(f"Initialized CheckpointManager with base dir: {self.checkpoint_dir}")

    def save_checkpoint(
        self,
        filepath: str,
        account_state: Dict[str, Any],
        trade_history: list,
        positions: list,
        checkpoint_date: datetime,
        metadata: Dict[str, Any],
        llm_cache: Optional[Dict[str, Any]] = None,
        include_llm_cache: bool = True
    ) -> None:
        """
        Save checkpoint to file

        Args:
            filepath: Path to save checkpoint (relative to checkpoint_dir or absolute)
            account_state: Account state dict
            trade_history: List of trade dicts
            positions: List of position dicts
            checkpoint_date: Timestamp of this checkpoint
            metadata: Additional metadata (model, dates, etc.)
            llm_cache: Optional LLM response cache
            include_llm_cache: Whether to include LLM cache in checkpoint
        """
        # Resolve filepath
        filepath_obj = Path(filepath)
        if not filepath_obj.is_absolute():
            # Only prepend checkpoint_dir if filepath doesn't already start with a results/ path
            if not str(filepath).startswith('results/'):
                filepath = self.checkpoint_dir / filepath
            else:
                # User provided full path from project root (e.g., results/checkpoints/...)
                filepath = filepath_obj

        # Create checkpoint data
        checkpoint = {
            "checkpoint_id": Path(filepath).stem,
            "created_at": datetime.now().isoformat(),
            "checkpoint_date": checkpoint_date.isoformat(),
            "account": account_state,
            "positions": positions,
            "trade_history": trade_history,
            "metadata": metadata,
        }

        # Add LLM cache if requested
        if include_llm_cache and llm_cache:
            checkpoint["llm_cache"] = llm_cache
            logger.info(f"Including LLM cache with {len(llm_cache)} entries")
        else:
            checkpoint["llm_cache"] = {}

        # Save to file (use pickle for complex objects)
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

        logger.info(f"Saved checkpoint to: {filepath}")
        logger.info(f"  Account value: ${account_state.get('account_value', 0):,.2f}")
        logger.info(f"  Positions: {len(positions)}")
        logger.info(f"  Trades: {len(trade_history)}")

    def load_checkpoint(
        self,
        filepath: str,
        use_llm_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint from file

        Args:
            filepath: Path to checkpoint file
            use_llm_cache: Whether to load LLM cache

        Returns:
            Checkpoint dict with all saved state
        """
        # Resolve filepath
        filepath_obj = Path(filepath)
        if not filepath_obj.is_absolute():
            # Only prepend checkpoint_dir if filepath doesn't already start with a results/ path
            if not str(filepath).startswith('results/'):
                filepath = self.checkpoint_dir / filepath
            else:
                # User provided full path from project root (e.g., results/checkpoints/...)
                filepath = filepath_obj

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        # Load checkpoint
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        logger.info(f"Loaded checkpoint from: {filepath}")
        logger.info(f"  Checkpoint ID: {checkpoint.get('checkpoint_id')}")
        logger.info(f"  Created: {checkpoint.get('created_at')}")
        logger.info(f"  Checkpoint date: {checkpoint.get('checkpoint_date')}")
        logger.info(f"  Account value: ${checkpoint['account'].get('account_value', 0):,.2f}")
        logger.info(f"  Positions: {len(checkpoint.get('positions', []))}")
        logger.info(f"  Trades: {len(checkpoint.get('trade_history', []))}")

        # Optionally clear LLM cache
        if not use_llm_cache:
            checkpoint["llm_cache"] = {}
            logger.info("  LLM cache disabled (forcing fresh API calls)")
        else:
            cache_size = len(checkpoint.get("llm_cache", {}))
            if cache_size > 0:
                logger.info(f"  LLM cache: {cache_size} entries")

        return checkpoint

    def save_metadata_json(self, filepath: str, checkpoint_data: Dict[str, Any]):
        """
        Save checkpoint metadata as JSON (for human readability)

        Args:
            filepath: Path to save JSON file
            checkpoint_data: Checkpoint dict
        """
        # Create human-readable summary
        summary = {
            "checkpoint_id": checkpoint_data.get("checkpoint_id"),
            "created_at": checkpoint_data.get("created_at"),
            "checkpoint_date": checkpoint_data.get("checkpoint_date"),
            "account": {
                "starting_capital": checkpoint_data["account"].get("starting_capital"),
                "account_value": checkpoint_data["account"].get("account_value"),
                "available_cash": checkpoint_data["account"].get("available_cash"),
                "total_return_percent": checkpoint_data["account"].get("total_return_percent"),
                "sharpe_ratio": checkpoint_data["account"].get("sharpe_ratio"),
                "total_fees_paid": checkpoint_data["account"].get("total_fees_paid"),
                "total_funding_paid": checkpoint_data["account"].get("total_funding_paid"),
            },
            "positions": [
                {
                    "symbol": pos.get("symbol"),
                    "quantity": pos.get("quantity"),
                    "entry_price": pos.get("entry_price"),
                    "current_price": pos.get("current_price"),
                    "leverage": pos.get("leverage"),
                    "unrealized_pnl": pos.get("unrealized_pnl"),
                }
                for pos in checkpoint_data.get("positions", [])
            ],
            "metadata": checkpoint_data.get("metadata", {}),
            "stats": {
                "total_trades": len(checkpoint_data.get("trade_history", [])),
                "open_positions": len(checkpoint_data.get("positions", [])),
                "llm_cache_entries": len(checkpoint_data.get("llm_cache", {})),
            }
        }

        # Resolve filepath
        filepath_obj = Path(filepath)
        if not filepath_obj.is_absolute():
            # Only prepend checkpoint_dir if filepath doesn't already start with a results/ path
            if not str(filepath).startswith('results/'):
                filepath = self.checkpoint_dir / filepath
            else:
                # User provided full path from project root (e.g., results/checkpoints/...)
                filepath = filepath_obj

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved checkpoint metadata to: {filepath}")

    def list_checkpoints(self) -> list:
        """
        List all available checkpoints

        Returns:
            List of checkpoint file paths
        """
        checkpoints = sorted(self.checkpoint_dir.glob("**/*.pkl"))
        logger.info(f"Found {len(checkpoints)} checkpoints in {self.checkpoint_dir}")
        return checkpoints

    def get_checkpoint_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get basic info about a checkpoint without fully loading it

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dict with basic checkpoint info
        """
        checkpoint = self.load_checkpoint(filepath, use_llm_cache=False)

        return {
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "created_at": checkpoint.get("created_at"),
            "checkpoint_date": checkpoint.get("checkpoint_date"),
            "account_value": checkpoint["account"].get("account_value"),
            "total_return_percent": checkpoint["account"].get("total_return_percent"),
            "sharpe_ratio": checkpoint["account"].get("sharpe_ratio"),
            "num_positions": len(checkpoint.get("positions", [])),
            "num_trades": len(checkpoint.get("trade_history", [])),
            "model": checkpoint.get("metadata", {}).get("model"),
        }
