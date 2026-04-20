import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.exceptions import EventFetchError, EventNotFoundError
from src.models import EventMetadata
from src.services.polymarket_service import PolymarketService


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_event_payload() -> dict:
    return {
        "id": "74949",
        "title": "Will GPT-5 launch before Q3 2025?",
        "description": "Resolves YES if OpenAI releases GPT-5 before July 1 2025.",
        "rules": "Official OpenAI announcement required.",
        "market_probability": 0.65,
        "liquidity": 120000.0,
        "ends_at": "2025-07-01T00:00:00Z",
    }


def _mock_response(status_code: int, body) -> MagicMock:
    """Build a mock requests.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = body
    mock.raise_for_status = MagicMock(
        side_effect=requests.HTTPError() if status_code >= 400 else None
    )
    return mock


# ── get_event_details ─────────────────────────────────────────────────────────

class TestGetEventDetails:
    @patch("src.services.polymarket_service.requests.get")
    def test_fetch_by_numeric_id_success(self, mock_get, raw_event_payload):
        mock_get.return_value = _mock_response(200, raw_event_payload)

        event = PolymarketService.get_event_details("74949")

        assert isinstance(event, EventMetadata)
        assert event.event_id == "74949"
        assert event.title == "Will GPT-5 launch before Q3 2025?"
        assert event.market_probability == 0.65

    @patch("src.services.polymarket_service.requests.get")
    def test_fetch_by_slug_success(self, mock_get, raw_event_payload):
        mock_get.return_value = _mock_response(200, [raw_event_payload])

        event = PolymarketService.get_event_details("will-gpt5-launch-q3-2025")

        assert isinstance(event, EventMetadata)
        assert event.event_id == "74949"

    @patch("src.services.polymarket_service.requests.get")
    def test_fetch_by_full_url_extracts_slug(self, mock_get, raw_event_payload):
        mock_get.return_value = _mock_response(200, [raw_event_payload])

        event = PolymarketService.get_event_details(
            "https://polymarket.com/event/will-gpt5-launch-q3-2025"
        )

        assert event is not None
        # Verify the slug was extracted correctly
        call_url = mock_get.call_args[0][0]
        assert "will-gpt5-launch-q3-2025" in call_url

    @patch("src.services.polymarket_service.requests.get")
    def test_http_404_raises_event_fetch_error(self, mock_get):
        mock_get.return_value = _mock_response(404, {})

        with pytest.raises(EventFetchError) as exc_info:
            PolymarketService.get_event_details("99999")

        assert exc_info.value.details["status_code"] == 404

    @patch("src.services.polymarket_service.requests.get")
    def test_http_500_raises_event_fetch_error(self, mock_get):
        mock_get.return_value = _mock_response(500, {})

        with pytest.raises(EventFetchError):
            PolymarketService.get_event_details("74949")

    @patch("src.services.polymarket_service.requests.get")
    def test_network_error_raises_event_fetch_error(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("DNS resolution failed")

        with pytest.raises(EventFetchError) as exc_info:
            PolymarketService.get_event_details("74949")

        assert "Network error" in exc_info.value.message

    @patch("src.services.polymarket_service.requests.get")
    def test_empty_slug_list_raises_not_found(self, mock_get):
        mock_get.return_value = _mock_response(200, [])

        with pytest.raises(EventNotFoundError) as exc_info:
            PolymarketService.get_event_details("nonexistent-slug")

        assert "nonexistent-slug" in exc_info.value.message

    @patch("src.services.polymarket_service.requests.get")
    def test_non_json_response_raises_event_fetch_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("No JSON")
        mock_get.return_value = mock_resp

        with pytest.raises(EventFetchError) as exc_info:
            PolymarketService.get_event_details("74949")

        assert "not valid JSON" in exc_info.value.message


# ── search_tech_events ────────────────────────────────────────────────────────

class TestSearchTechEvents:
    @patch("src.services.polymarket_service.requests.get")
    def test_returns_list_of_events(self, mock_get, raw_event_payload):
        mock_get.return_value = _mock_response(200, [raw_event_payload, raw_event_payload])

        events = PolymarketService.search_tech_events(limit=2)

        assert len(events) == 2
        assert all(isinstance(e, EventMetadata) for e in events)

    @patch("src.services.polymarket_service.requests.get")
    def test_network_failure_returns_empty_list(self, mock_get):
        """Discovery should degrade gracefully — never raise."""
        mock_get.side_effect = requests.ConnectionError("timeout")

        events = PolymarketService.search_tech_events()

        assert events == []

    @patch("src.services.polymarket_service.requests.get")
    def test_malformed_entry_is_skipped(self, mock_get, raw_event_payload):
        """A single bad entry should not crash the whole discovery."""
        bad_entry = {"id": None, "title": None}  # Will fail EventMetadata validation
        mock_get.return_value = _mock_response(200, [raw_event_payload, bad_entry])

        # Should not raise — bad entry is skipped
        events = PolymarketService.search_tech_events()
        assert len(events) >= 1
