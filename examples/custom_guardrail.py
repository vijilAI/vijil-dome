import re
from vijil_dome.detectors import (
    DetectionCategory,
    DetectionResult,
    DetectionMethod,
    register_method,
)

# Define our custom detector name
PHONE_NUMBER_DETECTOR = "phone-number-detector"

# Register with the Privacy category since we're protecting PII
@register_method(DetectionCategory.Privacy, PHONE_NUMBER_DETECTOR)
class PhoneNumberDetector(DetectionMethod):
    def __init__(self, 
                 block_international=True, 
                 strict_mode=False):
        super().__init__()
        self.block_international = block_international
        self.strict_mode = strict_mode
        
        # Define phone number patterns
        self.us_pattern = re.compile(
            r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10})'
        )
        
        self.international_pattern = re.compile(
            r'(\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9})'
        )
    
    async def detect(self, query_string: str) -> DetectionResult:
        # Check for US phone numbers
        us_matches = self.us_pattern.findall(query_string)
        
        # Check for international numbers if enabled
        intl_matches = []
        if self.block_international:
            intl_matches = self.international_pattern.findall(query_string)
        
        # Determine if we should flag this query
        flagged = bool(us_matches or intl_matches)
        
        # In strict mode, be more aggressive about potential numbers
        if self.strict_mode and not flagged:
            # Look for sequences that might be phone numbers
            digit_sequences = re.findall(r'\d{7,}', query_string)
            flagged = len(digit_sequences) > 0
        
        # Build metadata for logging and analysis
        metadata = {
            "type": type(self),
            "query_string": query_string,
            "us_phone_matches": us_matches,
            "international_matches": intl_matches,
            "strict_mode_triggered": self.strict_mode and bool(re.findall(r'\d{7,}', query_string)),
            "response_string": (
                "I can't process requests containing phone numbers to protect your privacy. "
                "Please remove any phone numbers and try again."
            ) if flagged else "Query processed successfully"
        }        
        return flagged, metadata


# Configure Dome with our custom guardrail
from vijil_dome import Dome

# Configure Dome with our custom guardrail
dome_config = {
    "pii-detection": {
        "type": "privacy",
        "methods": ["phone-number-detector"],
        "early-exit": True,
    },
    "input-guards": ["pii-detection"],
    "output-guards": ["pii-detection"],
    "input-early-exit": True
}
# Initialize Dome
dome = Dome(dome_config)


# Custom logic - block queries that are under a minimum number of characters or over a maximum number of characters
CUSTOM_LENGTH_DETECTOR = "custom-length-detector"
# You must register the detection method with one of five categories - Security, Moderation, Privacy, Integrity or Generic
@register_method(DetectionCategory.Security, CUSTOM_LENGTH_DETECTOR)
class CustomLengthDetector(DetectionMethod):
    def __init__(self, 
                 min_length = 10,
                 max_length = 1000):
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length
    # The detection method must be async, and must produce a DetectionResult
    async def detect(self, query_string: str) -> DetectionResult:
        flagged = len(query_string) < self.min_length or len(query_string) > self.max_length
        # The detection result is a tuple comprising of a boolean and a dictionary
        # The dictionary can contain any metadata you wish to record. We HIGHLY recommend including the original query string, type and response string
        return flagged, {
            "type": type(self),
            "length": len(query_string),
            "query_string": query_string,
            "response_string": "Query processed successfully"
            if flagged
            else "Query processed successfully",
        }

from vijil_dome import Dome
# from project.vijil_dome import Dome


# Initialize Dome
dome = Dome(dome_config)

import asyncio

async def test_phone_detector():
    # Test cases
    test_queries = [
        "Call me at (555) 123-4567 when you're ready",
        "My number is 555-123-4567",
        "Contact support at +1-800-555-0123",
        "I need help with my account balance",  # Safe query
        "The reference number is 1234567890",   # Might trigger strict mode
    ]
    
    for query in test_queries:
        try:
            # Process through Dome
            result = await dome.async_guard_input(query)
            print(f"Query: '{query}'")
            flagged = result.flagged
            response = result.response_string
            print(f"Result: {flagged=} {response=}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing '{query}': {e}")

# Run the test
asyncio.run(test_phone_detector())
