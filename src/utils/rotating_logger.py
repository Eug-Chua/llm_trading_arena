"""
Rotating logger that creates new log files after reaching a line limit
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime


class LineCountRotatingHandler(logging.Handler):
    """
    Custom handler that rotates log files based on line count
    Creates files named: logs_0.log, logs_1.log, logs_2.log, etc.
    """

    def __init__(self, log_dir: Path, max_lines: int = 5000, base_name: str = "logs"):
        """
        Initialize the rotating handler

        Args:
            log_dir: Directory to store log files
            max_lines: Maximum lines per file before rotation
            base_name: Base name for log files (default: "logs")
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.max_lines = max_lines
        self.base_name = base_name
        self.current_lines = 0
        self.current_file_num = 0

        # Find the latest log file number
        self._find_latest_file()

        # Open the current log file
        self.current_file = None
        self._open_new_file()

    def _find_latest_file(self):
        """Find the highest numbered log file and continue from there"""
        existing_files = list(self.log_dir.glob(f"{self.base_name}_*.log"))

        if existing_files:
            # Extract numbers from filenames
            numbers = []
            for f in existing_files:
                try:
                    num = int(f.stem.split('_')[-1])
                    numbers.append(num)
                except ValueError:
                    continue

            if numbers:
                self.current_file_num = max(numbers)

                # Count lines in the current file
                current_path = self.log_dir / f"{self.base_name}_{self.current_file_num}.log"
                if current_path.exists():
                    with open(current_path, 'r') as f:
                        self.current_lines = sum(1 for _ in f)

    def _open_new_file(self):
        """Open a new log file"""
        if self.current_file:
            self.current_file.close()

        filepath = self.log_dir / f"{self.base_name}_{self.current_file_num}.log"
        self.current_file = open(filepath, 'a')

    def _rotate(self):
        """Rotate to a new log file"""
        self.current_file_num += 1
        self.current_lines = 0
        self._open_new_file()

    def emit(self, record):
        """
        Emit a log record

        Args:
            record: LogRecord to emit
        """
        try:
            msg = self.format(record)

            # Check if we need to rotate
            if self.current_lines >= self.max_lines:
                self._rotate()

            # Write to current file
            self.current_file.write(msg + '\n')
            self.current_file.flush()

            self.current_lines += 1

        except Exception:
            self.handleError(record)

    def close(self):
        """Close the current log file"""
        if self.current_file:
            self.current_file.close()
        super().close()


def setup_rotating_logger(
    name: str,
    log_dir: str = "logs",
    max_lines: int = 5000,
    log_level: str = "INFO",
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with rotating file handler based on line count

    Args:
        name: Logger name
        log_dir: Directory to store log files
        max_lines: Maximum lines per file before rotation
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console_output: Whether to also output to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Add rotating file handler
    log_path = Path(__file__).parent.parent.parent / log_dir
    file_handler = LineCountRotatingHandler(
        log_dir=log_path,
        max_lines=max_lines,
        base_name="logs"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger
