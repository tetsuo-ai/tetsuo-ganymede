```markdown
# Tetsuo Ganymede - Jupiter Protocol Python SDK

A Python SDK for interacting with Jupiter Protocol on Solana. This SDK provides a simple interface to Jupiter's swap aggregator, DCA (Dollar Cost Average) trading, and price data services.

## Features

- Token Swaps
- Price Quotes
- DCA Trading
- Limit Orders
- Token Price Data
- Market Information
- Type Hints & Async Support

## Installation

Install the package using pip:

```bash
pip install tetsuo-ganymede
```

## Quick Start

```python
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from tetsuo_ganymede import Jupiter

# Initialize client
async def main():
    # Connect to Solana
    client = AsyncClient("https://api.mainnet-beta.solana.com")
    
    # Initialize with your wallet
    keypair = Keypair.from_bytes(your_private_key)
    jupiter = Jupiter(client, keypair)
    
    # Get a quote for swapping 1 SOL to USDC
    quote = await jupiter.quote(
        input_mint="So11111111111111111111111111111111111111112",  # SOL
        output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        amount=1_000_000_000  # 1 SOL (in lamports)
    )
    
    print(f"1 SOL = {int(quote['amount'])/1_000_000} USDC")

```

## Usage Examples

### Token Swapping

```python
# Get quote and execute swap
quote = await jupiter.quote(
    input_mint="So11111111111111111111111111111111111111112",  # SOL
    output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    amount=1_000_000_000,  # 1 SOL
    slippage_bps=50  # 0.5% slippage
)

# Execute the swap
tx = await jupiter.swap(
    input_mint="So11111111111111111111111111111111111111112",
    output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    amount=1_000_000_000,
    quote_response=quote
)
```

### DCA Trading

```python
# Create a DCA schedule to buy USDC with SOL
dca = await jupiter.dca.create_dca(
    input_mint="So11111111111111111111111111111111111111112",  # SOL
    output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    total_amount=10_000_000_000,  # 10 SOL total
    amount_per_cycle=1_000_000_000,  # 1 SOL per trade
    cycle_frequency=86400  # Daily trades
)

# Get DCA positions
positions = await jupiter.dca.get_dca_positions('active')

# Close a DCA position
await jupiter.dca.close_dca(dca_account="your_dca_account")
```

### Price Data

```python
# Get token prices
prices = await jupiter.get_token_price([
    "So11111111111111111111111111111111111111112",  # SOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
])

# Get tradeable tokens
tokens = await jupiter.get_tradeable_tokens(min_liquidity=1000000)  # $1M min liquidity

# Get tokens by tag
stables = await jupiter.get_tokens_by_tag("stable")
```

### Limit Orders

```python
# Place limit order
order = await jupiter.open_order(
    input_mint="So11111111111111111111111111111111111111112",  # SOL
    output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    in_amount=1_000_000_000,  # 1 SOL
    out_amount=20_000_000  # 20 USDC (price target)
)

# Get open orders
open_orders = await jupiter.query_open_orders(wallet_address)

# Cancel orders
await jupiter.cancel_orders(orders=[order_id])
```

## API Reference

### Jupiter Class

Main class providing access to Jupiter Protocol functionality.

#### Methods:

- `quote()` - Get swap route quotes
- `swap()` - Execute token swaps
- `get_token_price()` - Get token prices
- `get_tradeable_tokens()` - List tradeable tokens
- `get_tokens_by_tag()` - Filter tokens by tag
- `get_market_mints()` - Get market token pairs
- And more...

### DCA Class

Handles Dollar Cost Average trading functionality.

#### Methods:

- `create_dca()` - Create DCA schedule
- `close_dca()` - Close DCA position
- `get_dca_positions()` - List DCA positions
- `get_dca_trades()` - Get DCA trade history

## Error Handling

The SDK uses custom exceptions for different error cases:

```python
try:
    quote = await jupiter.quote(...)
except Exception as e:
    print(f"Error getting quote: {str(e)}")
```

## Rate Limits

Jupiter API has rate limits. Consider implementing retries and rate limiting in production:

```python
from asyncio import sleep

async def get_quote_with_retry(jupiter, retries=3):
    for i in range(retries):
        try:
            return await jupiter.quote(...)
        except Exception as e:
            if i == retries - 1:
                raise
            await sleep(1)
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

## Credits

Built on top of Jupiter Protocol - https://jup.ag

## Support

- GitHub Issues: [Create an issue](https://github.com/tetsuo-ai/tetsuo-ganymede/issues)
- Discord: [Join our community](discord.gg/tetsuo-ai)

## Changelog

### 0.1.0
- Initial release
- Basic swap functionality
- DCA trading support
- Price data endpoints

```