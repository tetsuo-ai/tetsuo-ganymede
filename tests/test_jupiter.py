import pytest
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from tetsuo_ganymede import Jupiter

# Test constants
TEST_RPC_URL = "https://api.mainnet-beta.solana.com"
TEST_TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
}

@pytest.fixture
async def jupiter():
    client = AsyncClient(TEST_RPC_URL)
    keypair = Keypair()
    jupiter = Jupiter(client, keypair)
    yield jupiter
    await client.close()

@pytest.mark.asyncio
async def test_quote(jupiter):
    quote = await jupiter.quote(
        input_mint=TEST_TOKENS["SOL"],
        output_mint=TEST_TOKENS["USDC"],
        amount=1_000_000_000  # 1 SOL
    )
    
    assert quote is not None
    assert "routePlan" in quote
    assert "amount" in quote
    assert "otherAmountThreshold" in quote
    assert int(quote["amount"]) > 0

@pytest.mark.asyncio
async def test_get_token_price(jupiter):
    prices = await jupiter.get_token_price(
        token_mints=[TEST_TOKENS["SOL"], TEST_TOKENS["USDC"]]
    )
    
    assert prices is not None
    assert len(prices) == 2
    assert all("price" in token_data for token_data in prices.values())

@pytest.mark.asyncio
async def test_get_tradeable_tokens(jupiter):
    tokens = await jupiter.get_tradeable_tokens(min_liquidity=1000000)  # $1M min liquidity
    
    assert tokens is not None
    assert len(tokens) > 0
    assert all("address" in token for token in tokens)
    assert all("symbol" in token for token in tokens)

@pytest.mark.asyncio
async def test_get_tokens_by_tag(jupiter):
    stable_tokens = await jupiter.get_tokens_by_tag("stable")
    
    assert stable_tokens is not None
    assert len(stable_tokens) > 0
    assert "USDC" in [token["symbol"] for token in stable_tokens]

@pytest.mark.asyncio
async def test_get_market_mints(jupiter):
    # Test with Orca USDC-SOL pool
    market = "8BnEgHoWFysVcuFFX7QztDmzuH8r5ZFvyP3sYwn1XTh6"
    mints = await jupiter.get_market_mints(market)
    
    assert mints is not None
    assert len(mints) == 2
    assert TEST_TOKENS["SOL"] in mints
    assert TEST_TOKENS["USDC"] in mints

@pytest.mark.asyncio
async def test_query_open_orders(jupiter):
    orders = await Jupiter.query_open_orders(str(jupiter.keypair.pubkey()))
    assert isinstance(orders, list)

@pytest.mark.asyncio
async def test_get_indexed_route_map(jupiter):
    route_map = await jupiter.get_indexed_route_map()
    
    assert route_map is not None
    assert "indexedRouteMap" in route_map
    assert "mintKeys" in route_map

@pytest.mark.asyncio
async def test_get_tokens_list(jupiter):
    tokens = await Jupiter.get_tokens_list()
    
    assert tokens is not None
    assert "tokens" in tokens
    assert len(tokens["tokens"]) > 0

# DCA Tests
@pytest.mark.asyncio
async def test_get_dca_positions(jupiter):
    positions = await jupiter.dca.get_dca_positions()
    assert isinstance(positions, list)

@pytest.mark.asyncio 
async def test_create_dca_errors(jupiter):
    with pytest.raises(Exception):
        await jupiter.dca.create_dca(
            input_mint="invalid_mint",
            output_mint=TEST_TOKENS["USDC"],
            total_amount=1000000,
            amount_per_cycle=100000,
            cycle_frequency=86400
        )

# Error case tests
@pytest.mark.asyncio
async def test_quote_invalid_token(jupiter):
    with pytest.raises(Exception):
        await jupiter.quote(
            input_mint="invalid_mint",
            output_mint=TEST_TOKENS["USDC"],
            amount=1000000
        )

@pytest.mark.asyncio
async def test_swap_insufficient_funds(jupiter):
    with pytest.raises(Exception):
        await jupiter.swap(
            input_mint=TEST_TOKENS["SOL"],
            output_mint=TEST_TOKENS["USDC"],
            amount=1000000000000  # Very large amount
        )

@pytest.mark.asyncio
async def test_get_token_price_invalid_mint(jupiter):
    with pytest.raises(Exception):
        await jupiter.get_token_price("invalid_mint")

# Edge case tests
@pytest.mark.asyncio
async def test_quote_zero_amount(jupiter):
    with pytest.raises(Exception):
        await jupiter.quote(
            input_mint=TEST_TOKENS["SOL"],
            output_mint=TEST_TOKENS["USDC"],
            amount=0
        )

@pytest.mark.asyncio
async def test_get_tokens_by_invalid_tag(jupiter):
    tokens = await jupiter.get_tokens_by_tag("invalid_tag")
    assert len(tokens) == 0

# Performance tests
@pytest.mark.asyncio
async def test_bulk_price_lookup(jupiter):
    # Test getting prices for 50 tokens at once
    tokens = await jupiter.get_tradeable_tokens(min_liquidity=1000000)
    token_mints = [token["address"] for token in tokens[:50]]
    
    prices = await jupiter.get_token_price(token_mints)
    assert len(prices) == len(token_mints)

@pytest.mark.asyncio
async def test_concurrent_requests(jupiter):
    import asyncio
    
    async def get_quote():
        return await jupiter.quote(
            input_mint=TEST_TOKENS["SOL"],
            output_mint=TEST_TOKENS["USDC"],
            amount=1_000_000_000
        )
    
    # Make 5 concurrent quote requests
    quotes = await asyncio.gather(*[get_quote() for _ in range(5)])
    assert len(quotes) == 5
    assert all(quote is not None for quote in quotes)
