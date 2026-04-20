from typing import Optional, List

import requests

from src.exceptions import EventFetchError, EventNotFoundError
from src.logger import get_logger
from src.models import EventMetadata

logger = get_logger(__name__)

_BASE_URL = "https://gamma-api.polymarket.com"
_TIMEOUT = 10  # seconds


class PolymarketService:

    @classmethod
    def get_event_details(cls, input_identifier: str) -> Optional[EventMetadata]:
        """
        Fetch a single event by ID, slug, or full Polymarket URL.

        Returns:
            EventMetadata on success, None if the event is not found.

        Raises:
            EventFetchError: on HTTP errors or unexpected API responses.
            EventNotFoundError: when the identifier resolves to no results.
        """
        # Normalise full URL → slug
        if "polymarket.com/event/" in input_identifier:
            input_identifier = (
                input_identifier.split("polymarket.com/event/")[1]
                .split("?")[0]
                .strip("/")
            )

        is_id = input_identifier.isdigit()
        url = (
            f"{_BASE_URL}/events/{input_identifier}"
            if is_id
            else f"{_BASE_URL}/events?slug={input_identifier}"
        )

        logger.info(
            "Fetching event from Polymarket.",
            extra={"identifier": input_identifier, "url": url},
        )

        try:
            response = requests.get(url, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            logger.error(
                "Network error fetching Polymarket event.",
                extra={"identifier": input_identifier, "error": str(exc)},
                exc_info=True,
            )
            raise EventFetchError(
                identifier=input_identifier, reason=f"Network error: {exc}"
            ) from exc

        if response.status_code != 200:
            logger.error(
                "Polymarket API returned non-200 status.",
                extra={"identifier": input_identifier, "status": response.status_code},
            )
            raise EventFetchError(
                identifier=input_identifier,
                status_code=response.status_code,
                reason=f"HTTP {response.status_code}",
            )

        try:
            data = response.json()
        except ValueError as exc:
            logger.error(
                "Polymarket API returned non-JSON body.",
                extra={"identifier": input_identifier},
                exc_info=True,
            )
            raise EventFetchError(
                identifier=input_identifier, reason="Response body is not valid JSON."
            ) from exc

        # Slug queries return a list
        if not is_id:
            if not isinstance(data, list) or len(data) == 0:
                logger.warning(
                    "No event found for slug.",
                    extra={"slug": input_identifier},
                )
                raise EventNotFoundError(identifier=input_identifier)
            data = data[0]

        event = EventMetadata(
            event_id=str(data.get("id")),
            title=data.get("title", ""),
            description=data.get("description", ""),
            resolution_rules=data.get("rules", ""),
            market_probability=data.get("market_probability"),
            liquidity=data.get("liquidity"),
            resolution_date=data.get("ends_at", ""),
        )

        logger.info(
            "Event fetched successfully.",
            extra={"event_id": event.event_id, "title": event.title[:60]},
        )
        return event

    @classmethod
    def search_tech_events(cls, limit: int = 10) -> List[EventMetadata]:
        """
        Discover trending AI/tech events on Polymarket.

        Returns an empty list (never raises) so the discover command
        degrades gracefully on API failures.
        """
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "search": "AI",
            "order": "liquidity",
            "ascending": "false",
        }

        logger.info("Discovering tech events.", extra={"limit": limit})

        try:
            response = requests.get(f"{_BASE_URL}/events", params=params, timeout=_TIMEOUT)
            response.raise_for_status()
            events_data = response.json()
        except Exception as exc:
            logger.error(
                "Failed to discover events.",
                extra={"error": str(exc)},
                exc_info=True,
            )
            return []

        events = []
        for data in events_data:
            try:
                events.append(
                    EventMetadata(
                        event_id=str(data.get("id")),
                        title=data.get("title", ""),
                        description=data.get("description", ""),
                        resolution_rules=data.get("rules", ""),
                        market_probability=data.get("market_probability"),
                        liquidity=data.get("liquidity"),
                        resolution_date=data.get("ends_at", ""),
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Skipping malformed event entry.",
                    extra={"error": str(exc), "data": str(data)[:200]},
                )

        logger.info("Event discovery complete.", extra={"count": len(events)})
        return events
