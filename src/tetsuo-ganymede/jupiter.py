"""
Jupiter Protocol Python SDK
"""
from typing import Dict, List, Any, Optional, Union

import base64
import json
import time
import struct
import httpx

from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.system_program import transfer, TransferParams
from solana.rpc.types import TxOpts
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Processed


class JupiterDCA:
    """Handles Jupiter's Dollar Cost Average (DCA) trading functionality.
    
    This class provides methods to:
    - Create and manage DCA trading schedules
    - Monitor trading progress and history
    - Configure custom trading parameters
    - Handle token account setup/cleanup
    """

    def __init__(
        self,
        async_client: AsyncClient,
        keypair: Keypair
    ):
        self.rpc = async_client
        self.keypair = keypair
        self.DCA_PROGRAM_ID = Pubkey.from_string("DCA265Vj8a9CEuX1eb1LWRnDT7uK6q1xMipnNyatn23M")

    async def create_dca(
        self,
        input_token: str,
        output_token: str,
        total_amount: int,
        amount_per_cycle: int,
        cycle_frequency: int,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        start_time: Optional[int] = None,
        close_wsol_account: bool = True
    ) -> Dict[str, Any]:
        """Create a new DCA trading schedule.
        
        Args:
            input_token: Token to sell (mint address)
            output_token: Token to buy (mint address)
            total_amount: Total amount to trade over time
            amount_per_cycle: Amount to trade each cycle
            cycle_frequency: Seconds between trades
            min_price: Minimum price to execute trade (optional)
            max_price: Maximum price to execute trade (optional)
            start_time: When to start trading (Unix timestamp)
            close_wsol_account: Auto close wrapped SOL account
            
        Returns:
            dict: DCA account info and creation tx hash
            
        Example:
            >>> # DCA 10 SOL into USDC over 10 days, 1 SOL per day
            >>> dca = await jupiter.dca.create_dca(
            >>>     input_token="So11111111111111111111111111111111111111112",
            >>>     output_token="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            >>>     total_amount=10_000_000_000,  # 10 SOL in lamports
            >>>     amount_per_cycle=1_000_000_000,  # 1 SOL per trade
            >>>     cycle_frequency=86400,  # 24 hours in seconds
            >>> )
        """
        try:
            # Generate unique DCA account ID
            uid = int(time.time())
            dca_account = await self._derive_dca_pubkey(
                input_token,
                output_token, 
                uid
            )

            # Setup token accounts if needed
            pre_instructions = []
            cleanup_instructions = []
            
            if input_token == WRAPPED_SOL_MINT:
                # Handle wrapped SOL setup
                token_account = get_associated_token_address(
                    self.keypair.pubkey(),
                    Pubkey.from_string(WRAPPED_SOL_MINT)
                )
                
                pre_instructions.extend([
                    # Create WSOL account if needed
                    create_associated_token_account(
                        self.keypair.pubkey(),
                        self.keypair.pubkey(),
                        Pubkey.from_string(WRAPPED_SOL_MINT)
                    ),
                    # Transfer SOL to be wrapped
                    transfer(
                        TransferParams(
                            from_pubkey=self.keypair.pubkey(),
                            to_pubkey=token_account,
                            lamports=total_amount
                        )
                    ),
                    # Sync native instruction
                    sync_native(
                        SyncNativeParams(
                            program_id=TOKEN_PROGRAM_ID,
                            account=token_account
                        )
                    )
                ])
                
                if close_wsol_account:
                    cleanup_instructions.append(
                        close_account(
                            CloseAccountParams(
                                account=token_account,
                                dest=self.keypair.pubkey(),
                                owner=self.keypair.pubkey(),
                                program_id=TOKEN_PROGRAM_ID
                            )
                        )
                    )

            # Build DCA open instruction
            accounts = {
                'dca': dca_account,
                'user': self.keypair.pubkey(),
                'inputMint': Pubkey.from_string(input_token),
                'outputMint': Pubkey.from_string(output_token),
                'userAta': get_associated_token_address(
                    self.keypair.pubkey(),
                    Pubkey.from_string(input_token)
                ),
                'inAta': get_associated_token_address(
                    dca_account,
                    Pubkey.from_string(input_token)
                ),
                'outAta': get_associated_token_address(
                    dca_account,
                    Pubkey.from_string(output_token)
                ),
                'systemProgram': Pubkey.from_string("11111111111111111111111111111111"),
                'tokenProgram': TOKEN_PROGRAM_ID,
                'associatedTokenProgram': ASSOCIATED_TOKEN_PROGRAM_ID
            }

            dca_ix = await self._build_dca_instruction(
                "openDca",
                accounts,
                {
                    'applicationIdx': uid,
                    'inAmount': total_amount,
                    'inAmountPerCycle': amount_per_cycle,
                    'cycleFrequency': cycle_frequency,
                    'minPrice': min_price,
                    'maxPrice': max_price,
                    'startAt': start_time or 0,
                    'closeWsolInAta': close_wsol_account
                }
            )

            # Build and send transaction
            tx = Transaction()
            for ix in pre_instructions:
                tx.add(ix)
            tx.add(dca_ix)
            for ix in cleanup_instructions:
                tx.add(ix)

            # Sign and send
            blockhash = await self.rpc.get_latest_blockhash()
            tx.recent_blockhash = blockhash.value.blockhash
            tx.sign(self.keypair)
            
            tx_hash = await self.rpc.send_transaction(
                tx,
                self.keypair,
                opts=TxOpts(skip_preflight=True)
            )

            return {
                'dca_account': str(dca_account),
                'transaction_hash': str(tx_hash.value),
                'uid': uid
            }

        except Exception as e:
            raise Exception(f"Error creating DCA: {str(e)}")

    async def close_dca(
        self,
        dca_account: str
    ) -> str:
        """Close a DCA trading schedule.
        
        Args:
            dca_account: DCA account public key
            
        Returns:
            str: Transaction hash
            
        Example:
            >>> tx_hash = await jupiter.dca.close_dca(
            >>>     "DCAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            >>> )
        """
        try:
            # Get DCA account data
            dca_data = await self.get_dca_data(dca_account)
            dca_pubkey = Pubkey.from_string(dca_account)

            # Build account addresses
            accounts = {
                'user': self.keypair.pubkey(),
                'dca': dca_pubkey,
                'inputMint': dca_data.input_mint,
                'outputMint': dca_data.output_mint,
                'inAta': dca_data.in_account,
                'outAta': dca_data.out_account,
                'userInAta': get_associated_token_address(
                    self.keypair.pubkey(),
                    dca_data.input_mint
                ),
                'userOutAta': get_associated_token_address(
                    self.keypair.pubkey(),
                    dca_data.output_mint
                ),
                'systemProgram': Pubkey.from_string("11111111111111111111111111111111"),
                'tokenProgram': TOKEN_PROGRAM_ID,
                'associatedTokenProgram': ASSOCIATED_TOKEN_PROGRAM_ID
            }

            # Build and send transaction
            close_ix = await self._build_dca_instruction(
                "closeDca",
                accounts,
                {}
            )

            tx = Transaction()
            tx.add(close_ix)

            blockhash = await self.rpc.get_latest_blockhash()
            tx.recent_blockhash = blockhash.value.blockhash
            tx.sign(self.keypair)
            
            tx_hash = await self.rpc.send_transaction(
                tx,
                self.keypair,
                opts=TxOpts(skip_preflight=True)
            )

            return str(tx_hash.value)

        except Exception as e:
            raise Exception(f"Error closing DCA: {str(e)}")

    async def get_dca_positions(
        self,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all DCA positions for the current wallet.
        
        Args:
            status: Filter by status ('active', 'completed', 'cancelled')
            
        Returns:
            list: List of DCA position details
            
        Example:
            >>> active_positions = await jupiter.dca.get_dca_positions('active')
        """
        try:
            wallet = str(self.keypair.pubkey())
            url = f"https://dca-api.jup.ag/user/{wallet}/dca"
            
            if status:
                status_map = {
                    'active': 0,
                    'completed': 1,
                    'cancelled': 2
                }
                url += f"?status={status_map[status]}"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                
                data = response.json()
                if not data.get('ok'):
                    raise ValueError(data.get('error', 'Unknown error'))
                    
                return data['data']['dcaAccounts']

        except Exception as e:
            raise Exception(f"Error fetching DCA positions: {str(e)}")

    async def get_dca_trades(
        self,
        dca_account: str
    ) -> List[Dict[str, Any]]:
        """Get trade history for a DCA position.
        
        Args:
            dca_account: DCA account public key
            
        Returns:
            list: List of executed trades
            
        Example:
            >>> trades = await jupiter.dca.get_dca_trades(
            >>>     "DCAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            >>> )
        """
        try:
            url = f"https://dca-api.jup.ag/dca/{dca_account}/fills"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                
                data = response.json()
                if not data.get('ok'):
                    raise ValueError(data.get('error', 'Unknown error'))
                    
                return data['data']['fills']

        except Exception as e:
            raise Exception(f"Error fetching DCA trades: {str(e)}")

    async def get_dca_data(
        self,
        dca_account: str
    ) -> Dict[str, Any]:
        """Get detailed data for a DCA account.
        
        Args:
            dca_account: DCA account public key
            
        Returns:
            dict: DCA account data and status including:
                - user: Owner public key
                - inputMint: Input token mint
                - outputMint: Output token mint
                - idx: Unique position ID
                - nextCycleAt: Next trade timestamp
                - inDeposited: Total amount deposited
                - inWithdrawn: Total amount withdrawn
                - outWithdrawn: Total output withdrawn
                - inUsed: Amount used in trades
                - outReceived: Amount received from trades
                - inAmountPerCycle: Trade size
                - cycleFrequency: Seconds between trades
                - inAccount: Input token account
                - outAccount: Output token account
                - createdAt: Creation timestamp
                
        Example:
            >>> details = await jupiter.dca.get_dca_data(
            >>>     "DCAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            >>> )
        """
        try:
            pubkey = Pubkey.from_string(dca_account)
            response = await self.rpc.get_account_info(pubkey)
            
            if not response.value:
                raise ValueError("DCA account not found")
                
            # Get raw account data
            data = response.value.data
            
            # Decode account data structure
            # Format from the DCA program:
            # pub struct Dca {
            #     pub user: Pubkey,            // 32 bytes
            #     pub input_mint: Pubkey,      // 32 bytes
            #     pub output_mint: Pubkey,     // 32 bytes
            #     pub idx: u64,                // 8 bytes
            #     pub next_cycle_at: i64,      // 8 bytes
            #     pub in_deposited: u64,       // 8 bytes
            #     pub in_withdrawn: u64,       // 8 bytes
            #     pub out_withdrawn: u64,      // 8 bytes
            #     pub in_used: u64,            // 8 bytes
            #     pub out_received: u64,       // 8 bytes
            #     pub in_amount_per_cycle: u64, // 8 bytes
            #     pub cycle_frequency: i64,     // 8 bytes
            #     pub in_account: Pubkey,      // 32 bytes
            #     pub out_account: Pubkey,     // 32 bytes
            #     pub created_at: i64,         // 8 bytes
            #     pub bump: u8,                // 1 byte
            # }
            
            offset = 0
            
            # Read public keys
            user = Pubkey(data[offset:offset+32])
            offset += 32
            input_mint = Pubkey(data[offset:offset+32])
            offset += 32
            output_mint = Pubkey(data[offset:offset+32])
            offset += 32
            
            # Read u64/i64 values
            idx = int.from_bytes(data[offset:offset+8], 'little')
            offset += 8
            next_cycle_at = int.from_bytes(data[offset:offset+8], 'little', signed=True)
            offset += 8
            in_deposited = int.from_bytes(data[offset:offset+8], 'little')
            offset += 8
            in_withdrawn = int.from_bytes(data[offset:offset+8], 'little')
            offset += 8
            out_withdrawn = int.from_bytes(data[offset:offset+8], 'little')
            offset += 8
            in_used = int.from_bytes(data[offset:offset+8], 'little')
            offset += 8
            out_received = int.from_bytes(data[offset:offset+8], 'little')
            offset += 8
            in_amount_per_cycle = int.from_bytes(data[offset:offset+8], 'little')
            offset += 8
            cycle_frequency = int.from_bytes(data[offset:offset+8], 'little', signed=True)
            offset += 8
            
            # Read token accounts
            in_account = Pubkey(data[offset:offset+32])
            offset += 32
            out_account = Pubkey(data[offset:offset+32])
            offset += 32
            
            # Read timestamps and bump
            created_at = int.from_bytes(data[offset:offset+8], 'little', signed=True)
            offset += 8
            bump = data[offset]
            
            return {
                "user": str(user),
                "inputMint": str(input_mint),
                "outputMint": str(output_mint),
                "idx": idx,
                "nextCycleAt": next_cycle_at,
                "inDeposited": in_deposited,
                "inWithdrawn": in_withdrawn,
                "outWithdrawn": out_withdrawn,
                "inUsed": in_used,
                "outReceived": out_received,
                "inAmountPerCycle": in_amount_per_cycle,
                "cycleFrequency": cycle_frequency,
                "inAccount": str(in_account),
                "outAccount": str(out_account),
                "createdAt": created_at,
                "bump": bump
            }
                
        except Exception as e:
            raise Exception(f"Error fetching DCA data: {str(e)}")

    async def _derive_dca_pubkey(
        self,
        input_token: str,
        output_token: str,
        uid: int
    ) -> Pubkey:
        """Derive DCA account address."""
        return Pubkey.find_program_address(
            [
                b"dca",
                self.keypair.pubkey().to_bytes(),
                Pubkey.from_string(input_token).to_bytes(),
                Pubkey.from_string(output_token).to_bytes(),
                uid.to_bytes(8, 'little')
            ],
            self.DCA_PROGRAM_ID
        )[0]

    async def _build_dca_instruction(
        self,
        method: str,
        accounts: Dict[str, Pubkey],
        args: Dict[str, Any]
    ) -> TransactionInstruction:
        """Build a DCA program instruction.
        
        Args:
            method: Instruction method name
            accounts: Dictionary of account pubkeys
            args: Method arguments
            
        Returns:
            TransactionInstruction: Ready to add to transaction
            
        This builds low-level Solana instructions for the DCA program.
        """
        # Define method layouts
        METHOD_LAYOUTS = {
            "openDca": {
                "prefix": bytes([0]),  # Method discriminator
                "layout": {
                    "application_idx": "u64",
                    "in_amount": "u64",
                    "in_amount_per_cycle": "u64",
                    "cycle_frequency": "i64",
                    "min_price": "optional[u64]",
                    "max_price": "optional[u64]",
                    "start_at": "optional[i64]",
                    "close_wsol_in_ata": "bool"
                }
            },
            "closeDca": {
                "prefix": bytes([1]),  # Method discriminator
                "layout": {}  # No args needed
            },
            "deposit": {
                "prefix": bytes([2]),
                "layout": {
                    "deposit_in": "u64"
                }
            },
            "withdraw": {
                "prefix": bytes([3]),
                "layout": {
                    "withdraw_params": {
                        "withdraw_amount": "u64",
                        "withdrawal": "enum[in,out]"
                    }
                }
            }
        }

        # Get method layout
        if method not in METHOD_LAYOUTS:
            raise ValueError(f"Unknown DCA method: {method}")
        layout = METHOD_LAYOUTS[method]

        # Build instruction data
        data = layout["prefix"]
        if layout["layout"]:
            # Pack arguments according to layout
            for field, field_type in layout["layout"].items():
                value = args.get(field)
                
                # Handle different field types
                if field_type == "u64":
                    data += int(value).to_bytes(8, "little")
                elif field_type == "i64":
                    data += int(value).to_bytes(8, "little", signed=True)
                elif field_type == "bool":
                    data += bytes([1 if value else 0])
                elif field_type.startswith("optional"):
                    base_type = field_type.split("[")[1][:-1]
                    if value is None:
                        data += bytes([0])  # None marker
                    else:
                        data += bytes([1])  # Some marker
                        if base_type == "u64":
                            data += int(value).to_bytes(8, "little")
                        elif base_type == "i64":
                            data += int(value).to_bytes(8, "little", signed=True)
                elif field_type.startswith("enum"):
                    options = field_type.split("[")[1][:-1].split(",")
                    if isinstance(value, str):
                        idx = options.index(value.lower())
                    else:
                        idx = int(value)
                    data += bytes([idx])
                elif isinstance(value, dict):
                    # Handle nested structures
                    for sub_field, sub_type in field_type.items():
                        sub_value = value.get(sub_field)
                        if sub_type == "u64":
                            data += int(sub_value).to_bytes(8, "little")
                        elif sub_type == "enum[in,out]":
                            data += bytes([0 if sub_value == "in" else 1])

        # Build account metas
        ACCOUNT_ROLES = {
            "openDca": {
                "writable": [
                    "dca", "user", "userAta", "inAta", "outAta"
                ],
                "signer": ["user"]
            },
            "closeDca": {
                "writable": [
                    "user", "dca", "inAta", "outAta", 
                    "userInAta", "userOutAta"
                ],
                "signer": ["user"]
            },
            "deposit": {
                "writable": [
                    "user", "dca", "inAta", "userInAta"
                ],
                "signer": ["user"]
            },
            "withdraw": {
                "writable": [
                    "user", "dca", "dcaAta", 
                    "userInAta", "userOutAta"
                ],
                "signer": ["user"]
            }
        }

        account_metas = []
        roles = ACCOUNT_ROLES[method]

        # Add accounts in correct order with proper meta flags
        for account_name, pubkey in accounts.items():
            account_metas.append({
                "pubkey": pubkey,
                "is_signer": account_name in roles["signer"],
                "is_writable": account_name in roles["writable"]
            })

        # Build and return instruction
        return TransactionInstruction(
            program_id=self.DCA_PROGRAM_ID,
            data=data,
            accounts=account_metas
        )
    
class Jupiter():
    
    ENDPOINT_APIS_URL = {
        "QUOTE": "https://quote-api.jup.ag/v6/quote?",
        "SWAP": "https://quote-api.jup.ag/v6/swap",
        "OPEN_ORDER": "https://jup.ag/api/limit/v1/createOrder",
        "CANCEL_ORDERS": "https://jup.ag/api/limit/v1/cancelOrders",
        "QUERY_OPEN_ORDERS": "https://jup.ag/api/limit/v1/openOrders?wallet=",
        "QUERY_ORDER_HISTORY": "https://jup.ag/api/limit/v1/orderHistory",
        "QUERY_TRADE_HISTORY": "https://jup.ag/api/limit/v1/tradeHistory",

        # Token endpoints
        "TOKEN": "https://token.jup.ag/v1/token",
        "MARKET_MINTS": "https://token.jup.ag/v1/market",
        "TOKENS": "https://token.jup.ag/v1/tokens",
    }
    
    def __init__(
        self,
        async_client: AsyncClient,
        keypair: Keypair,
        quote_api_url: str="https://quote-api.jup.ag/v6/quote?",
        swap_api_url: str="https://quote-api.jup.ag/v6/swap",
        open_order_api_url: str="https://jup.ag/api/limit/v1/createOrder",
        cancel_orders_api_url: str="https://jup.ag/api/limit/v1/cancelOrders",
        query_open_orders_api_url: str="https://jup.ag/api/limit/v1/openOrders?wallet=",
        query_order_history_api_url: str="https://jup.ag/api/limit/v1/orderHistory",
        query_trade_history_api_url: str="https://jup.ag/api/limit/v1/tradeHistory",
    ):
        self.dca = JupiterDCA(async_client, keypair)
        self.rpc = async_client
        self.keypair = keypair
        
        self.ENDPOINT_APIS_URL["QUOTE"] = quote_api_url
        self.ENDPOINT_APIS_URL["SWAP"] = swap_api_url
        self.ENDPOINT_APIS_URL["OPEN_ORDER"] = open_order_api_url
        self.ENDPOINT_APIS_URL["CANCEL_ORDERS"] = cancel_orders_api_url
        self.ENDPOINT_APIS_URL["QUERY_OPEN_ORDERS"] = query_open_orders_api_url
        self.ENDPOINT_APIS_URL["QUERY_ORDER_HISTORY"] = query_order_history_api_url
        self.ENDPOINT_APIS_URL["QUERY_TRADE_HISTORY"] = query_trade_history_api_url
    
    async def quote(
        self,
        input_mint: str,
        output_mint: str,  
        amount: int,
        slippage_bps: int = None,
        swap_mode: str = "ExactIn",
        only_direct_routes: bool = False,
        as_legacy_transaction: bool = False,
        exclude_dexes: list = None,
        max_accounts: int = None,
        platform_fee_bps: int = None,
        restrict_intermediate_tokens: bool = False
    ) -> dict:
        """Get the best swap route for a token trade pair sorted by largest output token amount.
        
        Args:
            Required:
                input_mint (str): Input token mint address
                output_mint (str): Output token mint address
                amount (int): The API takes in amount in integer and you have to factor in the decimals for each token
            Optional:
                slippage_bps (int): The slippage % in BPS. If the output token amount exceeds the slippage then the swap transaction will fail
                swap_mode (str): (ExactIn or ExactOut) Defaults to ExactIn
                only_direct_routes (bool): Default is False. Limits routing to single hop routes only
                as_legacy_transaction (bool): Default is False. Use legacy transaction format
                exclude_dexes (list): Default None. List of DEX names to exclude e.g. ['Aldrin','Saber']
                max_accounts (int): Find a route given a maximum number of accounts involved
                platform_fee_bps (int): Platform fee in basis points
                restrict_intermediate_tokens (bool): Restrict intermediate tokens to stable tokens
        
        Returns:
            dict: Best swap route response
            
        Example:
            >>> quote = await jupiter.quote(
            >>>     input_mint="So11111111111111111111111111111111111111112",
            >>>     output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", 
            >>>     amount=5_000_000
            >>> )
        """
        # Build base URL with required parameters
        quote_url = (
            f"{self.ENDPOINT_APIS_URL['QUOTE']}"
            f"inputMint={input_mint}"
            f"&outputMint={output_mint}"
            f"&amount={str(amount)}"
            f"&swapMode={swap_mode}"
            f"&onlyDirectRoutes={str(only_direct_routes).lower()}"
            f"&asLegacyTransaction={str(as_legacy_transaction).lower()}"
        )
        
        # Add optional parameters if specified
        if slippage_bps is not None:
            quote_url += f"&slippageBps={slippage_bps}"
        if exclude_dexes:
            quote_url += f"&excludeDexes={','.join(exclude_dexes)}"
        if max_accounts:
            quote_url += f"&maxAccounts={max_accounts}"
        if platform_fee_bps:
            quote_url += f"&platformFeeBps={platform_fee_bps}"
        if restrict_intermediate_tokens:
            quote_url += "&restrictIntermediateTokens=true"
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(quote_url, timeout=30.0)
                response.raise_for_status()
                quote_response = response.json()
                
                # Validate response has required fields
                if 'routePlan' not in quote_response:
                    raise ValueError("Invalid quote response: missing route plan")
                    
                return quote_response
                
        except httpx.RequestError as e:
            raise Exception(f"Network error fetching quote: {str(e)}")
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get('error', str(e))
            raise Exception(f"HTTP error fetching quote: {error_msg}")
        except ValueError as e:
            raise Exception(f"Invalid response: {str(e)}")
        except Exception as e:
            raise Exception(f"Error fetching quote: {str(e)}")

    async def swap(
        self,
        input_mint: str,
        output_mint: str,
        amount: int = 0,
        quote_response: dict = None,
        wrap_unwrap_sol: bool = True,
        slippage_bps: int = 1,
        swap_mode: str = "ExactIn",
        prioritization_fee_lamports: Optional[Dict[str, int]] = None,
        only_direct_routes: bool = False,
        as_legacy_transaction: bool = False,
        exclude_dexes: list = None,
        max_accounts: int = None,
        platform_fee_bps: int = None,
        use_shared_accounts: bool = True,
        destination_token_account: str = None,
        use_token_ledger: bool = False,  
        dynamic_compute_unit_limit: bool = True,
        skip_preflight: bool = False
    ) -> str:
        """Perform a swap.
        
        Args:
            Required:
                input_mint (str): Input token mint address
                output_mint (str): Output token mint address
                amount (int): Amount in lamports/raw units
            Optional:
                quote_response (dict): Response from quote API call
                wrap_unwrap_sol (bool): Auto wrap and unwrap SOL, default True
                slippage_bps (int): Slippage in basis points
                swap_mode (str): ExactIn or ExactOut, default ExactIn
                prioritization_fee_lamports (dict): Priority fee config:
                    {'priorityLevel': str, 'maxLamports': int} or {'jitoTipLamports': int}
                only_direct_routes (bool): Only use single hop routes
                as_legacy_transaction (bool): Use legacy transaction format
                exclude_dexes (list): List of DEX names to exclude
                max_accounts (int): Maximum accounts to use
                platform_fee_bps (int): Platform fee in basis points 
                use_shared_accounts (bool): Use shared program accounts
                destination_token_account (str): Custom destination token account
                use_token_ledger (bool): Use token ledger for input amount
                dynamic_compute_unit_limit (bool): Dynamic compute unit calculation
                skip_preflight (bool): Skip preflight transaction check
        
        Returns:
            str: Base64 encoded serialized transaction
            
        Example:
            >>> tx = await jupiter.swap(
            >>>     input_mint="So11111111111111111111111111111111111111112",
            >>>     output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            >>>     amount=1_000_000,
            >>>     slippage_bps=50
            >>> )
        """
        if quote_response is None:
            quote_response = await self.quote(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount,
                slippage_bps=slippage_bps,
                swap_mode=swap_mode,
                only_direct_routes=only_direct_routes,
                as_legacy_transaction=as_legacy_transaction,
                exclude_dexes=exclude_dexes,
                max_accounts=max_accounts,
                platform_fee_bps=platform_fee_bps
            )

        # Build swap request parameters
        swap_params = {
            "userPublicKey": str(self.keypair.pubkey()),
            "wrapAndUnwrapSol": wrap_unwrap_sol,
            "useSharedAccounts": use_shared_accounts,
            "dynamicComputeUnitLimit": dynamic_compute_unit_limit,
            "skipPreflight": skip_preflight,
            "useTokenLedger": use_token_ledger,
            "quoteResponse": quote_response
        }

        # Add optional parameters
        if prioritization_fee_lamports:
            swap_params["prioritizationFeeLamports"] = prioritization_fee_lamports
        if destination_token_account:
            swap_params["destinationTokenAccount"] = destination_token_account

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.ENDPOINT_APIS_URL['SWAP'],
                    json=swap_params,
                    timeout=30.0
                )
                response.raise_for_status()
                
                swap_response = response.json()
                if 'swapTransaction' not in swap_response:
                    raise ValueError("Missing swapTransaction in response")

                return swap_response['swapTransaction']

        except httpx.RequestError as e:
            raise Exception(f"Network error during swap: {str(e)}")
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get('error', str(e))
            raise Exception(f"HTTP error during swap: {error_msg}")
        except ValueError as e:
            raise Exception(f"Invalid swap response: {str(e)}")
        except Exception as e:
            raise Exception(f"Error performing swap: {str(e)}")

    async def get_swap_instructions(
        self,
        input_mint: str,
        output_mint: str,
        amount: int = 0,
        quote_response: dict = None,
        wrap_unwrap_sol: bool = True,
        slippage_bps: int = 1,
        swap_mode: str = "ExactIn",
        prioritization_fee_lamports: Optional[Dict[str, int]] = None,
        use_shared_accounts: bool = True,
        destination_token_account: str = None,
        use_token_ledger: bool = False,
        dynamic_compute_unit_limit: bool = True
    ) -> Dict[str, Any]:
        """Get detailed swap instructions for manual transaction building.
        
        Args:
            Required:
                input_mint (str): Input token mint address
                output_mint (str): Output token mint address
                amount (int): Amount in lamports/raw units
            Optional:
                quote_response (dict): Response from quote API call
                wrap_unwrap_sol (bool): Auto wrap and unwrap SOL
                slippage_bps (int): Slippage in basis points
                swap_mode (str): ExactIn or ExactOut
                prioritization_fee_lamports (dict): Priority fee config
                use_shared_accounts (bool): Use shared program accounts
                destination_token_account (str): Custom destination
                use_token_ledger (bool): Use token ledger
                dynamic_compute_unit_limit (bool): Dynamic CU limit
        
        Returns:
            dict: Containing:
                tokenLedgerInstruction: Optional token ledger setup
                computeBudgetInstructions: Compute budget setup
                otherInstructions: Additional setup (like Jito tips)
                setupInstructions: Account setup instructions
                swapInstruction: Main swap instruction
                cleanupInstruction: Optional cleanup (like unwrapping)
                addressLookupTableAddresses: For versioned transactions
            
        Example:
            >>> instructions = await jupiter.get_swap_instructions(
            >>>     input_mint="So11111111111111111111111111111111111111112",
            >>>     output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            >>>     amount=1_000_000
            >>> )
            >>> # Use with custom transaction building
            >>> transaction = VersionedTransaction(...)
            >>> transaction.add_instruction(instructions['swapInstruction'])
        """
        if quote_response is None:
            quote_response = await self.quote(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount,
                slippage_bps=slippage_bps,
                swap_mode=swap_mode
            )

        # Build request parameters
        swap_params = {
            "userPublicKey": str(self.keypair.pubkey()),
            "wrapAndUnwrapSol": wrap_unwrap_sol,
            "useSharedAccounts": use_shared_accounts,
            "dynamicComputeUnitLimit": dynamic_compute_unit_limit,
            "useTokenLedger": use_token_ledger,
            "quoteResponse": quote_response
        }

        if prioritization_fee_lamports:
            swap_params["prioritizationFeeLamports"] = prioritization_fee_lamports
        if destination_token_account:
            swap_params["destinationTokenAccount"] = destination_token_account

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.ENDPOINT_APIS_URL['SWAP_INSTRUCTIONS'],
                    json=swap_params,
                    timeout=30.0
                )
                response.raise_for_status()
                instructions = response.json()

                # Convert instruction data to proper format
                for key in ['tokenLedgerInstruction', 'swapInstruction', 'cleanupInstruction']:
                    if instructions.get(key):
                        instructions[key] = self._convert_instruction(instructions[key])

                # Convert arrays of instructions
                for key in ['computeBudgetInstructions', 'otherInstructions', 'setupInstructions']:
                    if instructions.get(key):
                        instructions[key] = [
                            self._convert_instruction(ix) for ix in instructions[key]
                        ]

                return instructions

        except httpx.RequestError as e:
            raise Exception(f"Network error fetching instructions: {str(e)}")
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get('error', str(e))
            raise Exception(f"HTTP error fetching instructions: {error_msg}")
        except Exception as e:
            raise Exception(f"Error fetching swap instructions: {str(e)}")

    def _convert_instruction(self, instruction_data: Dict[str, Any]) -> Instruction:
        """Convert API instruction format to Solana Instruction object."""
        try:
            return Instruction(
                accounts=[
                    AccountMeta(
                        pubkey=Pubkey.from_string(acct['pubkey']),
                        is_signer=acct['isSigner'],
                        is_writable=acct['isWritable']
                    )
                    for acct in instruction_data['accounts']
                ],
                program_id=Pubkey.from_string(instruction_data['programId']),
                data=base64.b64decode(instruction_data['data'])
            )
        except Exception as e:
            raise Exception(f"Error converting instruction: {str(e)}")

    async def open_order(
        self,
        input_mint: str,
        output_mint: str,
        in_amount: int=0,
        out_amount: int=0,
        expired_at: int=None
    ) -> dict:
        """Open an order.
        
        Args:
            Required:
                ``input_mint (str)``: Input token mint address\n
                ``output_mint (str)``: Output token mint address\n
                ``in_amount (int)``: The API takes in amount in integer and you have to factor in the decimals for each token by looking up the decimals for that token. For example, USDC has 6 decimals and 1 USDC is 1000000 in integer when passing it in into the API.\n
                ``out_amount (int)``: The API takes in amount in integer and you have to factor in the decimals for each token by looking up the decimals for that token. For example, USDC has 6 decimals and 1 USDC is 1000000 in integer when passing it in into the API.\n
            Optionals:
                ``expired_at (int)``: Deadline for when the limit order expires. It can be either None or Unix timestamp in seconds.
        Returns:
            ``dict``: transaction_data and signature2 in order to create the limit order.
            
        Example:
            >>> rpc_url = "https://neat-hidden-sanctuary.solana-mainnet.discover.quiknode.pro/2af5315d336f9ae920028bbb90a73b724dc1bbed/"
            >>> async_client = AsyncClient(rpc_url)
            >>> private_key_string = "tSg8j3pWQyx3TC2fpN9Ud1bS0NoAK0Pa3TC2fpNd1bS0NoASg83TC2fpN9Ud1bS0NoAK0P"
            >>> private_key = Keypair.from_bytes(base58.b58decode(private_key_string))
            >>> jupiter = Jupiter(async_client, private_key)
            >>> input_mint = "So11111111111111111111111111111111111111112"
            >>> output_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            >>> in_amount = 5_000_000
            >>> out_amount = 100_000
            >>> transaction_data = await jupiter.open_order(user_public_key, input_mint, output_mint, in_amount, out_amount)
            {
                'transaction_data': 'AgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgEGC5Qzg6Gwmq0Gtgp4+LWUVz0yQOAuHGNJAGTs0dcqEMVCBvqBKhFi2uRFEKYI4zPatxbdm7DylvnQUby9MexSmeAdsqhWUMQ86Ddz4+7pQFooE6wLglATS/YvzOVUNMOqnyAmC8Ioh9cSvEZniys4XY0OyEvxe39gSdHqlHWJQUPMn4prs0EwIc9JznmgzyMliG5PJTvaFYw75ssASGlB2gMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAImg/TLoYktlelMGKAi4mA0icnTD92092qSZhd3wNABMCv4fVqQvV1OYZ3a3bH43JpI5pIln+UAHnO1fyDJwCfIGm4hX/quBhPtof2NGGMA12sQ53BrrO1WYoPAAAAAAAQan1RcZLFxRIYzJTD1K8X9Y2u4Im6H9ROPb2YoAAAAABt324ddloZPZy+FGzut5rBy0he1fWzeROoz1hX7/AKmr+pT0gdwb1ZeE73qr11921UvCtCB3MMpBcLaiY8+u7QEHDAEABAMCCAIHBgUKCRmFbkqvcJ/1nxAnAAAAAAAAECcAAAAAAAAA',
                'signature2': Signature(
                    2Pip6gx9FLGVqmRqfAgwJ8HEuCY8ZbUbVERR18vHyxFngSi3Jxq8Vkpm74hS5zq7RAM6tqGUAkf3ufCBsxGXZrUC,)
            }
        """
        
        keypair = Keypair()
        transaction_parameters = {
            "owner": self.keypair.pubkey().__str__(),
            "inputMint": input_mint,
            "outputMint": output_mint,
            "outAmount": out_amount,
            "inAmount": in_amount,
            "base": keypair.pubkey().__str__()
        }
        if expired_at:
            transaction_parameters['expiredAt'] = expired_at
        transaction_data = httpx.post(url=self.ENDPOINT_APIS_URL['OPEN_ORDER'], json=transaction_parameters).json()['tx']
        raw_transaction = VersionedTransaction.from_bytes(base64.b64decode(transaction_data))
        signature2 = keypair.sign_message(message.to_bytes_versioned(raw_transaction.message))
        return {"transaction_data": transaction_data, "signature2": signature2}

    async def cancel_orders(
        self,
        orders: list=[]
    ) -> str:
        """Cancel open orders from a list (max. 10).
        
        Args:
            Required:!:
                ``orders (list)``: List of orders to be cancelled.
        Returns:
            ``str``: returns serialized transactions to cancel orders from https://jup.ag/api/limit/v1/cancelOrders
        
        Example:
            >>> rpc_url = "https://neat-hidden-sanctuary.solana-mainnet.discover.quiknode.pro/2af5315d336f9ae920028bbb90a73b724dc1bbed/"
            >>> async_client = AsyncClient(rpc_url)
            >>> private_key_string = "tSg8j3pWQyx3TC2fpN9Ud1bS0NoAK0Pa3TC2fpNd1bS0NoASg83TC2fpN9Ud1bS0NoAK0P"
            >>> private_key = Keypair.from_bytes(base58.b58decode(private_key_string))
            >>> jupiter = Jupiter(async_client, private_key)
            >>> list_orders = [item['publicKey'] for item in await jupiter.query_open_orders()] # Cancel all open orders
            >>> transaction_data = await jupiter.cancel_orders(orders=openOrders)
            AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAQIlDODobCarQa2Cnj4tZRXPTJA4C4cY0kAZOzR1yoQxUIklPdDonxNd5JDfdYoHE56dvNBQ1SLN90fFZxvVlzZr9DPwpfbd+ANTB35SSvHYVViD27UZR578oC2faxJea7y958guyGPhmEVKNR9GmJIjjuZU0VSr2/k044JZIRklkwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAr+H1akL1dTmGd2t2x+NyaSOaSJZ/lAB5ztX8gycAnyBpuIV/6rgYT7aH9jRhjANdrEOdwa6ztVmKDwAAAAAAEG3fbh12Whk9nL4UbO63msHLSF7V9bN5E6jPWFfv8AqW9ZjNTy3JS6YYFodCWqtWH80+eLPmN4igHrkYHIsdQfAQUHAQIAAwQHBghfge3wCDHfhA==
        """
        
        transaction_parameters = {
            "owner": self.keypair.pubkey().__str__(),
            "feePayer": self.keypair.pubkey().__str__(), 
            "orders": orders
        }
        transaction_data = httpx.post(url=self.ENDPOINT_APIS_URL['CANCEL_ORDERS'], json=transaction_parameters).json()['tx']
        return transaction_data

    async def query_open_orders(
        wallet_address: str,
        input_mint: str=None,
        output_mint: str=None
    ) -> list:
        """     
        Query open orders from self.keypair public address.
        
        Args:
            Required:
                ``wallet_address (str)``: Wallet address.
            Optionals:
                ``input_mint (str)``: Input token mint address.
                ``output_mint (str)``: Output token mint address.
        Returns:
            ``list``: returns open orders list from https://jup.ag/api/limit/v1/openOrders
            
        Example:
            >>> list_open_orders = await Jupiter.query_open_orders("AyWu89SjZBW1MzkxiREmgtyMKxSkS1zVy8Uo23RyLphX")
            [
                {   
                    'publicKey': '3ToRYxxMHN3CHkbqWHcbXBCBLNqmDeLoubGGfNKGSCDL',
                    'account': {
                        'maker': 'AyWu89SjZBW1MzkxiREmgtyMKxSkS1zVy8Uo23RyLphX',
                        'inputMint': 'So11111111111111111111111111111111111111112',
                        'outputMint': 'AGFEad2et2ZJif9jaGpdMixQqvW5i81aBdvKe7PHNfz3',
                        'oriInAmount': '10000',
                        'oriOutAmount': '10000',
                        'inAmount': '10000',
                        'outAmount': '10000',
                        'expiredAt': None,
                        'base': 'FghAhphJkhT74PXFQAz3QKqGVGA72Y2gUeVHU7QRw31c'
                    }
                }
            ]      
        """
        
        query_openorders_url = "https://jup.ag/api/limit/v1/openOrders?wallet=" + wallet_address
        if input_mint:
            query_openorders_url += "inputMint=" + input_mint
        if output_mint:
            query_openorders_url += "outputMint" + output_mint
            
        list_open_orders = httpx.get(query_openorders_url, timeout=Timeout(timeout=30.0)).json()
        return list_open_orders

    async def query_orders_history(
        wallet_address: str,
        cursor: int=None,
        skip: int=None,
        take: int=None
    ) -> list:
        """
        Query orders history from self.keypair public address.
        
        Args:
            Required:
                ``wallet_address (str)``: Wallet address.
            Optionals:
                ``cursor (int)``: Pointer to a specific result in the data set.
                ``skip (int)``: Number of records to skip from the beginning.
                ``take (int)``: Number of records to retrieve from the current position.
        Returns:
            ``list``: returns open orders list from https://jup.ag/api/limit/v1/orderHistory
            
        Example:
            >>> list_orders_history = await Jupiter.query_orders_history("AyWu89SjZBW1MzkxiREmgtyMKxSkS1zVy8Uo23RyLphX")
            [
                {
                    'id': 1639144,
                    'orderKey': '3ToRYxxMHN3CHkbqWHcbXBCBLNqmDeLoubGGfNKGSCDL',
                    'maker': 'AyWu89SjZBW1MzkxiREmgtyMKxSkS1zVy8Uo23RyLphX',
                    'inputMint': 'So11111111111111111111111111111111111111112',
                    'outputMint': 'AGFEad2et2ZJif9jaGpdMixQqvW5i81aBdvKe7PHNfz3',
                    'inAmount': '10000',
                    'oriInAmount': '10000',
                    'outAmount': '10000',
                    'oriOutAmount': '10000',
                    'expiredAt': None,
                    'state': 'Cancelled',
                    'createTxid': '4CYy8wZG2aRPctL9do7UBzaK9w4EDLJxGkkU1EEAx4LYNYW7j7Kyet2vL4q6cKK7HbJNHp6QXzLQftpTiDdhtyfL',
                    'cancelTxid': '83eN6Lm41t2VWUchm1T6hWX2qK3sf39XzPGbxV9s2WjBZfdUADQdRGg2Y1xAKn4igMJU1xRPCTgUhnm6qFUPWRc',
                    'updatedAt': '2023-12-18T15:55:30.617Z',
                    'createdAt': '2023-12-18T15:29:34.000Z'
                }
            ]
        """
        
        query_orders_history_url = "https://jup.ag/api/limit/v1/orderHistory" + "?wallet=" + wallet_address
        if cursor:
            query_orders_history_url += "?cursor=" + str(cursor)
        if skip:
            query_orders_history_url += "?skip=" + str(skip)
        if take:
            query_orders_history_url += "?take=" + str(take)
            
        list_orders_history = httpx.get(query_orders_history_url, timeout=Timeout(timeout=30.0)).json()
        return list_orders_history

    async def query_trades_history(
        wallet_address: str,
        input_mint: str=None,
        output_mint: str=None,
        cursor: int=None,
        skip: int=None,
        take: int=None
    ) -> list:
        """
        Query trades history from a public address.
        
        Args:
            Required:
                ``wallet_address (str)``: Wallet address.
            Optionals:
                ``input_mint (str)``: Input token mint address.
                ``output_mint (str)``: Output token mint address.
                ``cursor (int)``: Pointer to a specific result in the data set.
                ``skip (int)``: Number of records to skip from the beginning.
                ``take (int)``: Number of records to retrieve from the current position.
        Returns:
            ``list``: returns trades history list from https://jup.ag/api/limit/v1/tradeHistory
        
        Example:
            >>> list_trades_history = await Jupiter.query_trades_history("AyWu89SjZBW1MzkxiREmgtyMKxSkS1zVy8Uo23RyLphX")
            [
                {
                    'id': 10665592,
                    'inAmount':
                    '10000000',
                    'outAmount': '675870652',
                    'txid': '5rmA1S5MDAVdRYWeVgUWYFp6pYuy5vwrYpRJUJdhjoWnuuheeg1YwqK6P5H6u4tv99cUwQttSBYm6kjSNHJGENgb',
                    'updatedAt': '2023-12-13T15:39:04.800Z',
                    'createdAt': '2023-12-13T15:37:08.000Z',
                    'order': {
                        'id': 1278268,
                        'orderKey': '3bGykFCMWPNQDTRVBdKBbZuVHqNB5z5XaphkRHLWYmE5',
                        'inputMint': 'So11111111111111111111111111111111111111112',
                        'outputMint': '8XSsNvaKU9FDhYWAv7Yc7qSNwuJSzVrXBNEk7AFiWF69'
                    }
                }
            ]
        """
        
        query_tradeHistoryUrl = "https://jup.ag/api/limit/v1/tradeHistory" + "?wallet=" + wallet_address
        if input_mint:
            query_tradeHistoryUrl += "inputMint=" + input_mint
        if output_mint:
            query_tradeHistoryUrl += "outputMint" + output_mint
        if cursor:
            query_tradeHistoryUrl += "?cursor=" + cursor
        if skip:
            query_tradeHistoryUrl += "?skip=" + skip
        if take:
            query_tradeHistoryUrl += "?take=" + take
            
        tradeHistory = httpx.get(query_tradeHistoryUrl, timeout=Timeout(timeout=30.0)).json()
        return tradeHistory
    
    async def get_indexed_route_map(
    ) -> dict:
        """
        Retrieve an indexed route map for all the possible token pairs you can swap between.

        Returns:
            ``dict``: indexed route map for all the possible token pairs you can swap betwee from https://quote-api.jup.ag/v6/indexed-route-map
        
        Example:
            >>> indexed_route_map = await Jupiter.get_indexed_route_map()
        """
        
        indexed_route_map = httpx.get("https://quote-api.jup.ag/v6/indexed-route-map", timeout=Timeout(timeout=30.0)).json()
        return indexed_route_map

    async def get_tokens_list(
        list_type: str="strict",
        banned_tokens: bool=False
    ) -> dict:
        """
        The Jupiter Token List API is an open, collaborative, and dynamic token list to make trading on Solana more transparent and safer for users and developers.\n
        There are two types of list:\n
        ``strict``\n
            - Only tokens that are tagged "old-registry", "community", or "wormhole" verified.\n
            - No unknown and banned tokens.\n
        ``all``\n
            - Everything including unknown/untagged tokens that are picked up automatically.\n
            - It does not include banned tokens by default.\n
            - Often, projects notice that the token got banned and withdraw liquidity. As our lists are designed for trading, banned tokens that used to, but no longer meet our minimum liquidity requirements will not appear in this response.
        
        Args:
            Optionals:
                ``list_type (str)``: Default is "strict" (strict/all).
                ``banned_tokens (bool)``: Only if list_type is "all"
        Returns:
            ``dict``: indexed route map for all the possible token pairs you can swap betwee from https://token.jup.ag/{list_type}
        
        Example:
        >>> tokens_list = await Jupiter.get_tokens_list()
        """
        
        tokens_list_url = "https://token.jup.ag/"  + list_type
        if banned_tokens is True:
            tokens_list_url +=  "?includeBanned=true"
        tokens_list = httpx.get(tokens_list_url, timeout=Timeout(timeout=30.0)).json()
        return tokens_list
    
    
    async def get_token_stats_by_date(
        token: str,
        date: str,
    ) -> list:
        """Returns swap pairs for input token and output token
        
        Args:
            Required:
                ``token (str)``: Input token mint address.\n
                ``date (str)``: YYYY-MM-DD format date.\n
                
        Returns:
            ``list``: all swap pairs for input token and output token

        Example:
            >>> token_stats_by_date = await Jupiter.get_swap_pairs("B5mW68TkDewnKvWNc2trkmmdSRxcCjZz3Yd9BWxQTSRU", "2022-04-1")
        """
        token_stats_by_date_url = "https://stats.jup.ag/token-ledger/" + token + "/" + date
        token_stats_by_date = httpx.get(token_stats_by_date_url, timeout=Timeout(timeout=30.0)).json()
        return token_stats_by_date

    
    async def get_jupiter_stats(
        unit_of_time: str,
    ) -> dict:
        """Stats for the unit of time specified.
        
        Args:
            Required:
                ``unit_of_time (str)``: Unit of time: day/week/month
                
        Returns:
            ``dict``: stats for the unit of time specified.

        Example:
            >>> jupiter_stats = await Jupiter.get_jupiter_stats("day")
        """
        jupiter_stats_url = "https://stats.jup.ag/info/" + unit_of_time
        jupiter_stats = httpx.get(jupiter_stats_url, timeout=Timeout(timeout=30.0)).json()
        return jupiter_stats

    
    async def get_token_price(
        self,
        token_mints: Union[str, List[str]],
        vs_token: str = None,
        show_extra_info: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Get real-time price data for tokens.
        
        Args:
            token_mints: Single mint address or list of addresses
            vs_token: Optional token to price against (default USD)
            show_extra_info: Include confidence and depth info
        
        Returns:
            dict: Price data keyed by mint address containing:
                - price: Current price in USD or vs_token
                - id: Token mint address  
                - mintSymbol: Token symbol
                - vsToken: Denomination token
                - vsTokenSymbol: Denomination token symbol
                If show_extra_info=True, also includes:
                - lastUpdate: Unix timestamp
                - confidence: Price confidence level
                - priceChange24h: 24h price change
                - swapInPrice/swapOutPrice: Best available swap prices
                - liquidityLevel: Depth indicator
                
        Example:
            >>> # Get USDC price in SOL
            >>> price_data = await jupiter.get_token_price(
            >>>     "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            >>>     vs_token="So11111111111111111111111111111111111111112"
            >>> )
        """
        # Handle single mint or list
        if isinstance(token_mints, str):
            token_mints = [token_mints]
        
        # Build URL with parameters
        url = f"{self.ENDPOINT_APIS_URL['PRICE']}?ids={','.join(token_mints)}"
        if vs_token:
            url += f"&vsToken={vs_token}"
        if show_extra_info:
            url += "&showExtraInfo=true"
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                
                result = response.json()
                if 'data' not in result:
                    raise ValueError("Invalid price response format")
                    
                return result['data']
                
        except httpx.RequestError as e:
            raise Exception(f"Network error fetching price: {str(e)}")
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get('error', str(e))
            raise Exception(f"HTTP error fetching price: {error_msg}")
        except Exception as e:
            raise Exception(f"Error fetching price data: {str(e)}")

    async def get_token_stats(
        self,
        token_mint: str,
        days: int = 1
    ) -> Dict[str, Any]:
        """Get detailed trading stats for a token.
        
        Args:
            token_mint: Token mint address
            days: Number of days of data (1, 7, 30)
            
        Returns:  
            dict: Token statistics including:
                - price: Current price
                - priceChange: Price change % 
                - volume: Trading volume
                - tvl: Total value locked
                - marketCap: Market capitalization
                - holders: Number of holders
                - trades: Number of trades
                
        Example:
            >>> # Get 7-day SOL stats
            >>> stats = await jupiter.get_token_stats(
            >>>     "So11111111111111111111111111111111111111112",
            >>>     days=7
            >>> )
        """
        url = f"{self.ENDPOINT_APIS_URL['TOKEN_STATS']}/{token_mint}"
        params = {"days": days}
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=30.0)
                response.raise_for_status()
                return response.json()
                
        except httpx.RequestError as e:
            raise Exception(f"Network error fetching token stats: {str(e)}")
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get('error', str(e))
            raise Exception(f"HTTP error fetching token stats: {error_msg}")
        except Exception as e:
            raise Exception(f"Error fetching token stats: {str(e)}")
    
    async def get_token_info(
        self,
        token_mint: str
    ) -> Dict[str, Any]:
        """Get detailed information about a token.
        
        Args:
            token_mint: Token mint address
            
        Returns:
            dict: Token metadata including:
                - address: Token mint address
                - symbol: Token symbol
                - name: Token name
                - decimals: Token decimal places
                - logoURI: Token logo URL if available
                - tags: List of token tags
                - extensions: Additional metadata
                
        Example:
            >>> info = await jupiter.get_token_info(
            >>>     "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            >>> )
        """
        url = f"{self.ENDPOINT_APIS_URL['TOKEN']}/{token_mint}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                return response.json()
                
        except httpx.RequestError as e:
            raise Exception(f"Network error fetching token info: {str(e)}")
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get('error', str(e))
            raise Exception(f"HTTP error fetching token info: {error_msg}")
        except Exception as e:
            raise Exception(f"Error fetching token info: {str(e)}")

    async def get_market_mints(
        self, 
        market_address: str
    ) -> List[str]:
        """Get tokens involved in a specific market.
        
        Args:
            market_address: Market public key
            
        Returns:
            list: List of token mint addresses in the market
            
        Example:
            >>> mints = await jupiter.get_market_mints(
            >>>     "8BnEgHoWFysVcuFFX7QztDmzuH8r5ZFvyP3sYwn1XTh6"
            >>> )
        """
        url = f"{self.ENDPOINT_APIS_URL['MARKET_MINTS']}/{market_address}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                return response.json()
                
        except httpx.RequestError as e:
            raise Exception(f"Network error fetching market mints: {str(e)}")
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get('error', str(e))
            raise Exception(f"HTTP error fetching market mints: {error_msg}")
        except Exception as e:
            raise Exception(f"Error fetching market mints: {str(e)}")

    async def get_tradeable_tokens(
        self,
        min_liquidity: float = None,
        include_unwrapped_sol: bool = True
    ) -> List[Dict[str, Any]]:
        """Get list of all tokens tradeable via Jupiter.
        
        Args:
            min_liquidity: Optional minimum liquidity filter in USD
            include_unwrapped_sol: Include native SOL (default True)
            
        Returns:
            list: List of tradeable tokens with metadata
            
        Example:
            >>> tokens = await jupiter.get_tradeable_tokens(
            >>>     min_liquidity=10000  # $10k min liquidity
            >>> )
        """
        url = f"{self.ENDPOINT_APIS_URL['TOKENS']}/tradeable"
        params = {}
        if min_liquidity is not None:
            params['minLiquidity'] = min_liquidity
        if not include_unwrapped_sol:
            params['includeUnwrappedSol'] = 'false'
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=30.0)
                response.raise_for_status()
                return response.json()
                
        except httpx.RequestError as e:
            raise Exception(f"Network error fetching tradeable tokens: {str(e)}")
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get('error', str(e))
            raise Exception(f"HTTP error fetching tradeable tokens: {error_msg}")
        except Exception as e:
            raise Exception(f"Error fetching tradeable tokens: {str(e)}")

    async def get_new_tokens(
        self,
        min_days: int = None,
        max_days: int = None,
        min_liquidity: float = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get list of newly listed tokens on Jupiter.
        
        Args:
            min_days: Minimum days since listing
            max_days: Maximum days since listing  
            min_liquidity: Minimum liquidity in USD
            limit: Maximum number of tokens to return
            
        Returns:
            list: List of new tokens with metadata and market info
            
        Example:
            >>> # Get tokens listed in last 7 days with $100k+ liquidity
            >>> new_tokens = await jupiter.get_new_tokens(
            >>>     max_days=7,
            >>>     min_liquidity=100000
            >>> )
        """
        url = f"{self.ENDPOINT_APIS_URL['TOKENS']}/new"
        params = {'limit': min(limit, 1000)}
        
        if min_days is not None:
            params['minDays'] = min_days
        if max_days is not None:
            params['maxDays'] = max_days
        if min_liquidity is not None:
            params['minLiquidity'] = min_liquidity
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=30.0)
                response.raise_for_status()
                return response.json()
                
        except httpx.RequestError as e:
            raise Exception(f"Network error fetching new tokens: {str(e)}")
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get('error', str(e))
            raise Exception(f"HTTP error fetching new tokens: {error_msg}")
        except Exception as e:
            raise Exception(f"Error fetching new tokens: {str(e)}")

    async def get_tokens_by_tag(
        self,
        tags: Union[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Get tokens filtered by tags.
        
        Args:
            tags: Single tag or list of tags (e.g. 'stable', 'wrapped')
            
        Returns:
            list: List of tokens matching any of the tags
            
        Example:
            >>> # Get all stable tokens
            >>> stables = await jupiter.get_tokens_by_tag('stable')
            >>> # Get wrapped and bridged tokens
            >>> wrapped = await jupiter.get_tokens_by_tag(['wrapped', 'bridged'])
        """
        if isinstance(tags, str):
            tags = [tags]
            
        tag_list = ','.join(tags)
        url = f"{self.ENDPOINT_APIS_URL['TOKENS']}/tagged/{tag_list}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                return response.json()
                
        except httpx.RequestError as e:
            raise Exception(f"Network error fetching tagged tokens: {str(e)}")
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get('error', str(e))
            raise Exception(f"HTTP error fetching tagged tokens: {error_msg}")
        except Exception as e:
            raise Exception(f"Error fetching tagged tokens: {str(e)}")

    
    async def program_id_to_label(
    ) -> dict:
        """Returns a dict, which key is the program id and value is the label.\n
        This is used to help map error from transaction by identifying the fault program id.\n
        With that, we can use the exclude_dexes or dexes parameter for swap.

        Returns:
            ``dict``: program_id and label

        Example:
            >>> program_id_to_label_list = await Jupiter.program_id_to_label()
        """
        program_id_to_label_list = httpx.get("https://quote-api.jup.ag/v6/program-id-to-label", timeout=Timeout(timeout=30.0)).json()
        return program_id_to_label_list
