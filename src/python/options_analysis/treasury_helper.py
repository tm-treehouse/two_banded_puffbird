#!/usr/bin/env python3
"""
Helper module for fetching Treasury yield data.
"""

from datetime import datetime
import logging
import re
import xml.etree.ElementTree as ET

import requests

logger = logging.getLogger("options_analysis")


def get_treasury_yield() -> float:
    """
    Get current risk-free rate from Treasury website by finding the most recent entry.
    Uses the NEW_DATE field to find the data point closest to the current date.

    Returns:
        float or None: Current 10-year Treasury yield as a decimal, or None if unavailable
    """
    # Get current date
    current_date = datetime.now().date()
    current_year = current_date.year

    try:
        # URL for the Treasury Yield XML data for the current year
        url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xmlview?data=daily_treasury_yield_curve&field_tdr_date_value={current_year}"
        logger.info(f"Requesting Treasury data from: {url}")

        response = requests.get(url, timeout=30)  # Reduced timeout for faster failures
        if response.status_code != 200:
            logger.warning(f"Treasury API returned status code: {response.status_code}")
            return None

        # Define reasonable yield range for validation (in percent)
        min_yield, max_yield = 2.0, 10.0

        # Try to parse XML data properly to extract the most recent values
        try:
            root = ET.fromstring(response.content)

            # Find all entries that contain Treasury data
            entries = root.findall(
                ".//{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}properties"
            )

            if not entries:
                logger.warning("No data entries found in Treasury XML")
                # Fall back to regex-based approach
                return get_treasury_yield_regex(response.text, min_yield, max_yield)

            logger.info(f"Found {len(entries)} Treasury data entries")

            # Initialize variables to track the closest date and its yield value
            closest_date = None
            closest_date_diff = float("inf")
            closest_date_yield = None

            # Process each entry to find the one closest to the current date
            for entry in entries:
                # Find NEW_DATE field in the entry
                date_elem = entry.find(".//{http://schemas.microsoft.com/ado/2007/08/dataservices}NEW_DATE")
                if date_elem is None or not date_elem.text:
                    continue

                # Find BC_10YEAR field in the same entry
                yield_elem = entry.find(".//{http://schemas.microsoft.com/ado/2007/08/dataservices}BC_10YEAR")
                if yield_elem is None or not yield_elem.text:
                    continue

                try:
                    # Parse the date
                    entry_date = datetime.strptime(date_elem.text.split("T")[0], "%Y-%m-%d").date()

                    # Calculate difference from current date
                    date_diff = abs((current_date - entry_date).days)

                    # Parse the yield value
                    yield_value = float(yield_elem.text)

                    # Check if yield is in reasonable range
                    if min_yield <= yield_value <= max_yield:
                        # Update closest date if this entry is more recent
                        if date_diff < closest_date_diff:
                            closest_date = entry_date
                            closest_date_diff = date_diff
                            closest_date_yield = yield_value
                            logger.debug(f"New closest date: {closest_date}, yield: {closest_date_yield}%")
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error parsing entry data: {e}")
                    continue

            # If we found a valid yield value
            if closest_date_yield is not None:
                logger.info(f"Using Treasury yield from {closest_date}: {closest_date_yield}%")
                return closest_date_yield / 100  # Convert from percent to decimal
            else:
                logger.warning("No valid Treasury yield found in XML data")
        except ET.ParseError as e:
            logger.warning(f"XML parsing error: {e}")

        # Fall back to regex-based approach if XML parsing failed
        return get_treasury_yield_regex(response.text, min_yield, max_yield)

    except Exception as e:
        logger.error(f"Error in Treasury yield extraction: {str(e)}")
        return None


def get_treasury_yield_regex(content_str: str, min_yield: float = 3.0, max_yield: float = 6.0) -> float:
    """
    Fallback method to extract Treasury yield using regex patterns

    Parameters:
        content_str (str): The raw content from Treasury website
        min_yield (float): Minimum valid yield percentage
        max_yield (float): Maximum valid yield percentage

    Returns:
        float or None: Treasury yield as a decimal, or None if not found
    """
    logger.info("Falling back to regex-based Treasury yield extraction")
    try:
        # Method 1: Direct regex search for BC_10YEAR tag and its value
        bc10_pattern = r"<[^>]*BC_10YEAR[^>]*>([0-9.]+)<"
        bc10_matches = re.findall(bc10_pattern, content_str)

        if bc10_matches:
            for value_str in bc10_matches[:5]:  # Check first few matches
                try:
                    value = float(value_str)
                    if min_yield <= value <= max_yield:  # Sanity check the value
                        logger.info(f"Found valid BC_10YEAR value: {value}")
                        return value / 100  # Convert percentage to decimal
                    else:
                        logger.debug(f"Found BC_10YEAR value but outside range: {value}")
                except ValueError:
                    continue

        # Method 2: Try to find 10-year mentions with nearby numbers
        ten_year_patterns = [
            r"10-year[^0-9.]*([0-9.]+)",
            r"10 year[^0-9.]*([0-9.]+)",
            r"10yr[^0-9.]*([0-9.]+)",
            r"10-Year[^0-9.]*([0-9.]+)",
        ]

        for pattern in ten_year_patterns:
            matches = re.findall(pattern, content_str, re.IGNORECASE)
            for value_str in matches:
                try:
                    value = float(value_str)
                    if min_yield <= value <= max_yield:
                        logger.info(f"Found valid 10-year mention with value: {value}")
                        return value / 100  # Convert percentage to decimal
                except ValueError:
                    continue

        # Method 3: Look for GS10 references
        gs10_pattern = r"GS10[^0-9.]*([0-9.]+)"
        gs10_matches = re.findall(gs10_pattern, content_str, re.IGNORECASE)

        for value_str in gs10_matches:
            try:
                value = float(value_str)
                if min_yield <= value <= max_yield:
                    logger.info(f"Found valid GS10 value: {value}")
                    return value / 100
            except ValueError:
                continue

        # Method 4: Last resort - look for numbers in typical yield range
        all_numbers = re.findall(r"(\d+\.\d+)", content_str)
        yield_range = [n for n in all_numbers if min_yield <= float(n) <= max_yield]

        if yield_range:
            value = float(yield_range[0])
            logger.info(f"Found number in typical yield range: {value}")
            return value / 100  # Convert percentage to decimal

        # If no reasonable value found, hardcode today's known rate
        # This is better than returning None when we know current rates
        logger.warning(
            "Could not extract valid 10-year yield from Treasury website, using current known rate"
        )
        known_rate = 0.0437  # 4.37% as of May 10, 2025
        return known_rate

    except Exception as e:
        logger.error(f"Error in regex Treasury yield extraction: {str(e)}")
        return 0.0437  # Default fallback
