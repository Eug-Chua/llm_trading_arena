#!/usr/bin/env python3
"""
Web scraper for nof1.ai to extract model prompts and decision history

This script uses Selenium to handle JavaScript-rendered content and
extract prompt examples from all Alpha Arena models.

Requirements:
    pip install selenium webdriver-manager

Usage:
    python scripts/scrape_nof1_prompts.py
    python scripts/scrape_nof1_prompts.py --model deepseek-chat-v3.1
    python scripts/scrape_nof1_prompts.py --output data/prompts/ --no-headless
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
except ImportError:
    print("ERROR: Selenium not installed. Please run:")
    print("  pip install selenium webdriver-manager")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.rotating_logger import setup_rotating_logger

logger = setup_rotating_logger(
    name="nof1_scraper",
    log_dir="logs",
    max_lines=5000,
    log_level="INFO",
    console_output=True
)

MODELS = [
    'deepseek-chat-v3.1',
    'gpt-5',
    'claude-sonnet-4-5',
    'gemini-2.5-pro',
    'grok-4',
    'qwen3-max'
]


class Nof1Scraper:
    """Web scraper for nof1.ai Alpha Arena"""

    def __init__(self, headless: bool = True):
        """
        Initialize the scraper

        Args:
            headless: Run browser in headless mode (no GUI)
        """
        logger.info("Initializing Nof1 scraper")

        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')

        # Initialize Chrome driver
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.wait = WebDriverWait(self.driver, 20)

        logger.info("Browser initialized successfully")

    def __del__(self):
        """Clean up browser on exit"""
        if hasattr(self, 'driver'):
            self.driver.quit()

    def scrape_homepage_modelchat(self, filter_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Scrape all model decisions from the homepage MODELCHAT tab

        Args:
            filter_model: Optional model ID to filter results

        Returns:
            Dictionary with all decisions grouped by model
        """
        logger.info(f"Navigating to nof1.ai homepage")

        try:
            # Load homepage
            self.driver.get("https://nof1.ai/")
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            logger.info("Homepage loaded, waiting for MODELCHAT button...")

            # Wait longer for tabs to appear and click MODELCHAT tab
            try:
                # Wait up to 20 seconds for the button to be present AND clickable
                modelchat_button = self.wait.until(
                    EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'MODELCHAT')]"))
                )
                logger.info("MODELCHAT button found, waiting for it to be clickable...")

                # Give it a moment to finish rendering
                time.sleep(2)

                # Now click it
                modelchat_button = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'MODELCHAT')]"))
                )
                modelchat_button.click()
                logger.info("Clicked MODELCHAT tab")
                time.sleep(3)  # Wait for content to render after click
            except TimeoutException:
                logger.warning("MODELCHAT button not found within 20 seconds")
                # Try to continue anyway - content might already be visible
                pass

            # Extract all decisions
            result = {
                'scrape_timestamp': datetime.now().isoformat(),
                'filter_model': filter_model,
                'total_decisions': 0,
                'decisions_by_model': {}
            }

            # Try multiple selectors for decision containers
            selectors = [
                "//div[contains(@class, 'group cursor-pointer')]",
                "//div[contains(@class, 'group') and contains(@class, 'cursor-pointer')]",
                "//div[@class='group cursor-pointer rounded-lg border border-gray-800 bg-black/50 p-4 transition-all hover:border-blue-500']",
                "//div[contains(@class, 'rounded-lg border')]//span[contains(@class, 'terminal-text')]/../..",
            ]

            decision_containers = []
            for selector in selectors:
                decision_containers = self.driver.find_elements(By.XPATH, selector)
                if decision_containers:
                    logger.info(f"Found {len(decision_containers)} decision containers using selector: {selector}")
                    break

            if not decision_containers:
                logger.warning("No decision containers found with any selector")
                logger.info("Saving page source for debugging...")
                result['page_source_sample'] = self.driver.page_source[:5000]  # First 5000 chars

            for idx, container in enumerate(decision_containers):
                try:
                    # Check if browser window is still open
                    try:
                        self.driver.current_url
                    except Exception:
                        logger.error("Browser window was closed by user. Saving extracted data so far.")
                        break

                    decision = self._extract_decision_from_container(container, idx)

                    if decision and decision.get('model_name'):
                        model_name = decision['model_name']

                        # Filter by model if specified
                        if filter_model and filter_model.lower() not in model_name.lower():
                            continue

                        # Group by model
                        if model_name not in result['decisions_by_model']:
                            result['decisions_by_model'][model_name] = []

                        result['decisions_by_model'][model_name].append(decision)
                        result['total_decisions'] += 1

                        logger.info(f"Extracted decision {idx} from {model_name}")

                except Exception as e:
                    logger.error(f"Error processing container {idx}: {e}")
                    # If it's a window closed error, break the loop
                    if "no such window" in str(e) or "target window already closed" in str(e):
                        logger.error("Browser window closed. Stopping extraction.")
                        break
                    continue

            logger.info(f"Successfully extracted {result['total_decisions']} decisions")
            return result

        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            raise

    def _extract_decision_from_container(self, container, idx: int) -> Optional[Dict[str, Any]]:
        """
        Extract decision data from a single container element

        Args:
            container: Selenium WebElement
            idx: Index for logging

        Returns:
            Dictionary with decision data or None
        """
        decision = {
            'decision_index': idx,
            'model_name': None,
            'timestamp': None,
            'summary': None,
            'user_prompt': None,
            'chain_of_thought': None,
            'trading_decisions': None
        }

        # Extract model name
        try:
            # Model name is in a span with class "terminal-text text-sm font-semibold"
            model_elem = container.find_element(
                By.XPATH,
                ".//span[contains(@class, 'terminal-text') and contains(@class, 'text-sm') and contains(@class, 'font-semibold')]"
            )
            decision['model_name'] = model_elem.text.strip()
        except NoSuchElementException:
            logger.debug(f"Model name not found for decision {idx}")
            return None

        # Extract timestamp
        try:
            timestamp_elem = container.find_element(
                By.XPATH,
                ".//span[contains(@class, 'text-gray-500')]"
            )
            decision['timestamp'] = timestamp_elem.text.strip()
        except NoSuchElementException:
            pass

        # Extract summary
        try:
            summary_elem = container.find_element(
                By.XPATH,
                ".//div[contains(@class, 'terminal-text') and contains(@class, 'text-xs') and contains(@class, 'leading-relaxed')]"
            )
            decision['summary'] = summary_elem.text.strip()
        except NoSuchElementException:
            pass

        # Extract hidden sections (USER_PROMPT, CHAIN_OF_THOUGHT, TRADING_DECISIONS)
        # These are in the page source but hidden by CSS
        try:
            # Find the hidden expanded section
            expanded_section = container.find_element(
                By.XPATH,
                ".//div[contains(@class, 'hidden') and contains(@class, 'transition-all')]"
            )

            # Extract USER_PROMPT
            try:
                # Find the div containing USER_PROMPT label, then get the content div
                user_prompt_section = expanded_section.find_element(
                    By.XPATH,
                    ".//span[text()='USER_PROMPT']"
                )
                # Navigate to the content div: up to button, then to sibling div with ml-4, then to the content div
                user_prompt_div = user_prompt_section.find_element(
                    By.XPATH,
                    "ancestor::button/following-sibling::div[@class='ml-4']/div[contains(@class, 'rounded border border-border bg-surface-elevated')]"
                )
                # Use textContent to get all text including from hidden elements
                decision['user_prompt'] = user_prompt_div.get_attribute('textContent').strip()
            except NoSuchElementException:
                pass

            # Extract CHAIN_OF_THOUGHT
            try:
                cot_section = expanded_section.find_element(
                    By.XPATH,
                    ".//span[text()='CHAIN_OF_THOUGHT']"
                )
                cot_div = cot_section.find_element(
                    By.XPATH,
                    "ancestor::button/following-sibling::div[@class='ml-4']/div[contains(@class, 'rounded border border-border bg-surface-elevated')]"
                )
                decision['chain_of_thought'] = cot_div.get_attribute('textContent').strip()
            except NoSuchElementException:
                pass

            # Extract TRADING_DECISIONS
            try:
                td_section = expanded_section.find_element(
                    By.XPATH,
                    ".//span[text()='TRADING_DECISIONS']"
                )
                # TRADING_DECISIONS might have a different structure (grid of cards), so get parent div
                td_div = td_section.find_element(
                    By.XPATH,
                    "ancestor::button/following-sibling::div[@class='ml-4']"
                )
                decision['trading_decisions'] = td_div.get_attribute('textContent').strip()
            except NoSuchElementException:
                pass

        except NoSuchElementException:
            logger.debug(f"Expanded section not found for decision {idx}")

        return decision


def main():
    parser = argparse.ArgumentParser(description="Scrape nof1.ai for model prompts")
    parser.add_argument(
        '--model',
        type=str,
        help='Filter for specific model (e.g., deepseek-chat-v3.1, gpt-5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/prompts',
        help='Output directory for scraped data'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        default=True,
        help='Run browser in headless mode'
    )
    parser.add_argument(
        '--no-headless',
        action='store_false',
        dest='headless',
        help='Run browser with GUI (for debugging)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("NOF1.AI PROMPT SCRAPER")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Headless mode: {args.headless}")
    if args.model:
        logger.info(f"Filtering for model: {args.model}")

    try:
        scraper = Nof1Scraper(headless=args.headless)

        # Scrape homepage
        result = scraper.scrape_homepage_modelchat(filter_model=args.model)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if args.model:
            # Save filtered results
            output_file = output_dir / f"{args.model}_{timestamp}.json"
        else:
            # Save all results
            output_file = output_dir / f"all_models_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info("="*80)
        logger.info("SCRAPING COMPLETE")
        logger.info("="*80)
        logger.info(f"Total decisions extracted: {result['total_decisions']}")
        logger.info(f"Models found: {list(result['decisions_by_model'].keys())}")
        for model, decisions in result['decisions_by_model'].items():
            logger.info(f"  {model}: {len(decisions)} decisions")
        logger.info(f"Results saved to {output_file}")

    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error during scraping: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
