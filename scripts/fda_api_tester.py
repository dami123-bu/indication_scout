"""Standalone script to hit the openFDA FAERS API and inspect raw responses."""

import asyncio
import json
import logging

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://api.fda.gov/drug/event.json"
DRUG_NAME = "metformin"


async def get_top_reactions(session: aiohttp.ClientSession, drug_name: str, limit: int = 10) -> dict:
    """Count mode: top adverse reactions by frequency."""
    params = {
        "search": f'patient.drug.openfda.generic_name:"{drug_name}"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": limit,
    }
    async with session.get(BASE_URL, params=params) as resp:
        logger.info("get_top_reactions status: %s", resp.status)
        return await resp.json()


async def get_events(session: aiohttp.ClientSession, drug_name: str, limit: int = 5) -> dict:
    """Search mode: individual adverse event reports."""
    params = {
        "search": f'patient.drug.openfda.generic_name:"{drug_name}"',
        "limit": limit,
    }
    async with session.get(BASE_URL, params=params) as resp:
        logger.info("get_events status: %s", resp.status)
        return await resp.json()


async def main() -> None:
    async with aiohttp.ClientSession() as session:
        logger.info("--- get_top_reactions for '%s' ---", DRUG_NAME)
        top = await get_top_reactions(session, DRUG_NAME)
        print(json.dumps(top, indent=2))

        logger.info("--- get_events for '%s' ---", DRUG_NAME)
        events = await get_events(session, DRUG_NAME)
        print(json.dumps(events, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
