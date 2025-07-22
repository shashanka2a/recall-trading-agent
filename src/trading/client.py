"""
Recall API client for trading operations
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
import aiohttp

from config.settings import TradingConfig
from src.core.models import Portfolio, Position, MarketData, TradeOrder, TradeResult
from src.core.enums import OrderStatus
from src.core.exceptions import APIError, RateLimitError, ValidationError

logger = logging.getLogger(__name__)


class RecallClient:
    """Enhanced Recall API client with rate limiting and error handling"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter: Dict[str, float] = {}
        self._request_count = 0
        self._last_reset = time.time()
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                "Authorization": f"Bearer {self.config.recall_api_key}",
                "User-Agent": "RecallTradingAgent/2.0",
                "Content-Type": "application/json"
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit_check(self, endpoint: str) -> None:
        """Implement rate limiting to prevent API abuse"""
        now = time.time()
        
        # Reset counter every minute
        if now - self._last_reset > 60:
            self._request_count = 0
            self._last_reset = now
        
        # Check global rate limit
        if self._request_count >= 50:  # Conservative limit
            sleep_time = 60 - (now - self._last_reset)
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                self._request_count = 0
                self._last_reset = time.time()
        
        # Check endpoint-specific rate limit
        if endpoint in self.rate_limiter:
            time_since_last = now - self.rate_limiter[endpoint]
            if time_since_last < self.config.rate_limit_delay:
                sleep_time = self.config.rate_limit_delay - time_since_last
                await asyncio.sleep(sleep_time)
        
        self.rate_limiter[endpoint] = time.time()
        self._request_count += 1
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Tuple[bool, Dict]:
        """Make HTTP request with error handling"""
        
        if not self.session:
            raise APIError("Client session not initialized")
        
        await self._rate_limit_check(endpoint)
        
        url = f"{self.config.recall_api_url}{endpoint}"
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                json=data,
                params=params
            ) as response:
                
                response_data = await response.json()
                
                if response.status == 200:
                    logger.debug(f"✅ {method} {endpoint} - Success")
                    return True, response_data
                elif response.status == 429:
                    raise RateLimitError("Rate limit exceeded")
                else:
                    error_msg = response_data.get("error", f"HTTP {response.status}")
                    logger.error(f"❌ {method} {endpoint} - {error_msg}")
                    return False, response_data
                    
        except aiohttp.ClientTimeout:
            logger.error(f"⏰ {method} {endpoint} - Timeout")
            return False, {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"❌ {method} {endpoint} - {str(e)}")
            return False, {"error": str(e)}
    
    async def get_portfolio(self) -> Portfolio:
        """Fetch current portfolio state"""
        success, data = await self._make_request("GET", "/balances")
        
        if not success:
            logger.error("Failed to fetch portfolio")
            return Portfolio(cash=0.0)
        
        try:
            cash = float(data.get("cash", 0.0))
            positions = {}
            
            # Parse positions
            for symbol, balance in data.get("positions", {}).items():
                if float(balance) > 0:
                    # Get current price for position valuation
                    price_data = await self.get_market_data(symbol)
                    current_price = price_data.price if price_data else 0.0
                    
                    positions[symbol] = Position(
                        symbol=symbol,
                        quantity=float(balance),
                        avg_price=current_price,  # Simplified - would need trade history
                        current_price=current_price,
                        unrealized_pnl=0.0  # Will be calculated after price update
                    )
            
            portfolio = Portfolio(
                cash=cash,
                positions=positions,
                target_allocations=self.config.target_allocations
            )
            
            portfolio.update_total_value()
            logger.info(f"📊 Portfolio value: ${portfolio.total_value:.2f}")
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Error parsing portfolio data: {e}")
            return Portfolio(cash=0.0)
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for symbol"""
        success, data = await self._make_request("GET", f"/price/{symbol}")
        
        if not success:
            logger.warning(f"Failed to fetch price for {symbol}")
            return None
        
        try:
            return MarketData(
                symbol=symbol,
                price=float(data.get("price", 0)),
                volume=float(data.get("volume", 0)),
                change_24h=float(data.get("change_24h", 0)),
                timestamp=time.time(),
                high_24h=data.get("high_24h"),
                low_24h=data.get("low_24h"),
                bid=data.get("bid"),
                ask=data.get("ask"),
                chain=data.get("chain", "")
            )
        except Exception as e:
            logger.error(f"Error parsing market data for {symbol}: {e}")
            return None
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get market data for multiple symbols efficiently"""
        results = {}
        
        # Use asyncio.gather for concurrent requests
        tasks = [self.get_market_data(symbol) for symbol in symbols]
        market_data_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, market_data in zip(symbols, market_data_list):
            if isinstance(market_data, MarketData):
                results[symbol] = market_data
            else:
                logger.warning(f"Failed to get data for {symbol}")
        
        return results
    
    async def get_quote(self, order: TradeOrder) -> Optional[Dict]:
        """Get trading quote for order"""
        params = {
            "from_token": order.from_token,
            "to_token": order.to_token,
            "amount": order.amount
        }
        
        success, data = await self._make_request("GET", "/quote", params=params)
        
        if success:
            logger.debug(f"💱 Quote: {order.amount} {order.from_token} → "
                        f"{data.get('expected_output', 0)} {order.to_token}")
            return data
        else:
            logger.error(f"Failed to get quote for {order.from_token} → {order.to_token}")
            return None
    
    async def execute_trade(self, order: TradeOrder) -> TradeResult:
        """Execute a trade order"""
        
        # Validate order
        if not order.validate():
            return TradeResult(
                order=order,
                success=False,
                error_message="Invalid order parameters"
            )
        
        # Check if dry run mode
        if self.config.dry_run:
            logger.info(f"🧪 DRY RUN: {order.amount:.4f} {order.from_token} → {order.to_token}")
            return TradeResult(
                order=order,
                success=True,
                executed_amount=order.amount,
                actual_output=order.expected_output or order.amount,
                actual_slippage=0.001  # Simulated slippage
            )
        
        # Get quote first for validation
        quote = await self.get_quote(order)
        if not quote:
            return TradeResult(
                order=order,
                success=False,
                error_message="Failed to get trading quote"
            )
        
        # Check slippage
        expected_output = float(quote.get("expected_output", 0))
        slippage = float(quote.get("slippage", 0))
        
        if slippage > order.max_slippage:
            return TradeResult(
                order=order,
                success=False,
                error_message=f"Slippage {slippage:.3f} exceeds limit {order.max_slippage:.3f}"
            )
        
        # Execute the trade
        trade_data = {
            "from_token": order.from_token,
            "to_token": order.to_token,
            "amount": order.amount
        }
        
        logger.info(f"🔄 Executing: {order.amount:.4f} {order.from_token} → {order.to_token}")
        logger.info(f"   Expected: {expected_output:.4f} {order.to_token} (slippage: {slippage:.3f})")
        
        success, data = await self._make_request("POST", "/execute_trade", data=trade_data)
        
        if success:
            logger.info(f"✅ Trade executed successfully!")
            return TradeResult(
                order=order,
                success=True,
                executed_amount=order.amount,
                actual_output=data.get("actual_output", expected_output),
                actual_slippage=data.get("actual_slippage", slippage),
                transaction_id=data.get("transaction_id"),
                fee=data.get("fee")
            )
        else:
            error_msg = data.get("error", "Unknown error")
            logger.error(f"❌ Trade failed: {error_msg}")
            return TradeResult(
                order=order,
                success=False,
                error_message=error_msg
            )
    
    async def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get recent trade history"""
        params = {"limit": limit}
        success, data = await self._make_request("GET", "/trades", params=params)
        
        if success:
            return data.get("trades", [])
        else:
            logger.error("Failed to fetch trade history")
            return []
    
    async def get_competition_status(self) -> Optional[Dict]:
        """Get current competition status"""
        success, data = await self._make_request("GET", "/competition/status")
        
        if success:
            return data
        else:
            logger.error("Failed to fetch competition status")
            return None
    
    async def get_leaderboard(self) -> Optional[Dict]:
        """Get competition leaderboard"""
        success, data = await self._make_request("GET", "/competition/leaderboard")
        
        if success:
            return data
        else:
            logger.error("Failed to fetch leaderboard")
            return None
    
    async def health_check(self) -> bool:
        """Check API health and connectivity"""
        try:
            success, data = await self._make_request("GET", "/health")
            if success:
                logger.info("✅ API health check passed")
                return True
            else:
                logger.warning("⚠️ API health check failed")
                return False
        except Exception as e:
            logger.error(f"❌ API health check error: {e}")
            return False
    
    def get_request_stats(self) -> Dict[str, int]:
        """Get API request statistics"""
        return {
            "total_requests": self._request_count,
            "time_since_reset": int(time.time() - self._last_reset),
            "endpoints_accessed": len(self.rate_limiter)
        }