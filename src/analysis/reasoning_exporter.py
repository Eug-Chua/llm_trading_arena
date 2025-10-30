"""
Reasoning Exporter

Exports LLM reasoning (chain of thought + signals) from checkpoints to JSON.
"""

import json
from pathlib import Path
from datetime import datetime


def export_llm_reasoning(checkpoint: dict, checkpoint_path: str, metadata: dict = None):
    """
    Export LLM reasoning from checkpoint to JSON file

    Args:
        checkpoint: Loaded checkpoint dict with llm_cache
        checkpoint_path: Path to the checkpoint file
        metadata: Optional metadata dict from checkpoint

    Returns:
        Path to the exported reasoning JSON file (or None if no LLM responses)
    """
    checkpoint_path = Path(checkpoint_path)
    llm_cache = checkpoint.get('llm_cache', {})

    if not llm_cache:
        print("⚠️  No LLM responses found in checkpoint")
        return None

    # Generate filename matching checkpoint: checkpoint_name_reasoning.json
    reasoning_file = checkpoint_path.parent / f"{checkpoint_path.stem}_reasoning.json"

    print(f"Exporting LLM reasoning to {reasoning_file}...")

    # Sort responses by timestamp
    sorted_responses = sorted(llm_cache.items(), key=lambda x: x[1].get('timestamp', ''))

    # Build structured data
    reasoning_data = {
        "backtest_info": {
            "checkpoint_id": checkpoint_path.stem,
            "start_date": metadata.get('start_date', 'unknown') if metadata else 'unknown',
            "end_date": metadata.get('end_date', 'unknown') if metadata else 'unknown',
            "model": metadata.get('model', 'unknown') if metadata else 'unknown',
            "coins": metadata.get('coins', []) if metadata else [],
            "interval": metadata.get('interval') if metadata else None,
            "total_iterations": len(sorted_responses),
            "exported_at": datetime.now().isoformat()
        },
        "iterations": []
    }

    # Export each iteration
    for i, (prompt_hash, response_data) in enumerate(sorted_responses, 1):
        timestamp = response_data.get('timestamp', 'unknown')
        model = response_data.get('model', 'unknown')
        agent_response = response_data.get('response')

        # Get raw response text
        raw_text = ""
        if agent_response:
            if hasattr(agent_response, 'raw_response'):
                raw_text = agent_response.raw_response
            elif isinstance(agent_response, dict):
                raw_text = agent_response.get('raw_response', '')
            else:
                raw_text = str(agent_response)

        # Extract signals if available
        signals = []
        if agent_response:
            trade_signals_data = None
            if hasattr(agent_response, 'trade_signals'):
                trade_signals_data = agent_response.trade_signals
            elif isinstance(agent_response, dict):
                trade_signals_data = agent_response.get('trade_signals')

            if trade_signals_data:
                # Handle both dict and list formats
                signal_list = trade_signals_data.values() if isinstance(trade_signals_data, dict) else trade_signals_data

                for signal in signal_list:
                    # Handle both object attributes and dict keys
                    def get_attr(obj, key):
                        return getattr(obj, key, None) if hasattr(obj, key) else (obj.get(key) if isinstance(obj, dict) else None)

                    signals.append({
                        "coin": get_attr(signal, 'coin'),
                        "signal": get_attr(signal, 'signal'),
                        "quantity": get_attr(signal, 'quantity'),
                        "leverage": get_attr(signal, 'leverage'),
                        "stop_loss": get_attr(signal, 'stop_loss'),
                        "profit_target": get_attr(signal, 'profit_target'),
                        "confidence": get_attr(signal, 'confidence'),
                        "risk_usd": get_attr(signal, 'risk_usd'),
                        "invalidation_condition": get_attr(signal, 'invalidation_condition'),
                        "close_reason": get_attr(signal, 'close_reason')
                    })

        iteration_data = {
            "iteration": i,
            "timestamp": timestamp,
            "model": model,
            "raw_response": raw_text,
            "signals": signals
        }

        reasoning_data["iterations"].append(iteration_data)

    # Save to JSON file
    with open(reasoning_file, 'w') as f:
        json.dump(reasoning_data, f, indent=2)

    print(f"✓ Exported {len(sorted_responses)} LLM responses to {reasoning_file}")

    return str(reasoning_file)
