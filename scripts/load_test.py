"""
Production load testing script (Phase 4).

Tests:
- Concurrent requests
- Rate limiting behavior
- Multi-tenant isolation
- Response times under load
"""

import asyncio
import aiohttp
import time
from typing import List, Dict
import statistics


async def make_request(
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    query: str,
) -> Dict:
    """Make a single API request."""
    start_time = time.time()
    
    headers = {"X-API-Key": api_key}
    data = {"query": query}
    
    try:
        async with session.post(url, json=data, headers=headers) as response:
            status = response.status
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "success": status == 200,
                "status": status,
                "duration_ms": duration_ms,
            }
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return {
            "success": False,
            "status": 0,
            "duration_ms": duration_ms,
            "error": str(e),
        }


async def load_test(
    url: str,
    api_key: str,
    num_requests: int = 100,
    concurrency: int = 10,
):
    """
    Run load test.
    
    Args:
        url: API endpoint URL
        api_key: API key for authentication
        num_requests: Total number of requests
        concurrency: Number of concurrent requests
    """
    print(f"\n{'='*80}")
    print(f"LOAD TEST")
    print(f"{'='*80}")
    print(f"URL: {url}")
    print(f"Total requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"{'='*80}\n")
    
    queries = [
        "What is RAG?",
        "Explain vector embeddings",
        "How does similarity search work?",
        "What is LangChain?",
        "Describe LlamaIndex",
    ]
    
    results: List[Dict] = []
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        # Create batches
        for i in range(0, num_requests, concurrency):
            batch_size = min(concurrency, num_requests - i)
            
            tasks = [
                make_request(
                    session,
                    url,
                    api_key,
                    queries[j % len(queries)],
                )
                for j in range(batch_size)
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Progress
            completed = len(results)
            print(f"Progress: {completed}/{num_requests} ({completed/num_requests*100:.1f}%)")
            
            # Small delay between batches
            if i + batch_size < num_requests:
                await asyncio.sleep(0.1)
        
        total_duration = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    durations = [r["duration_ms"] for r in successful]
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Requests/second: {len(results)/total_duration:.2f}")
    
    if durations:
        print(f"\nResponse Times:")
        print(f"  Min: {min(durations):.2f}ms")
        print(f"  Max: {max(durations):.2f}ms")
        print(f"  Mean: {statistics.mean(durations):.2f}ms")
        print(f"  Median: {statistics.median(durations):.2f}ms")
        print(f"  P95: {statistics.quantiles(durations, n=20)[18]:.2f}ms")
        print(f"  P99: {statistics.quantiles(durations, n=100)[98]:.2f}ms")
    
    # Status code distribution
    status_codes = {}
    for r in results:
        status = r["status"]
        status_codes[status] = status_codes.get(status, 0) + 1
    
    print(f"\nStatus Codes:")
    for status, count in sorted(status_codes.items()):
        print(f"  {status}: {count}")
    
    if failed:
        print(f"\nSample Errors:")
        for r in failed[:5]:
            if "error" in r:
                print(f"  - {r['error']}")


async def rate_limit_test(
    url: str,
    api_key: str,
    requests_per_minute: int = 60,
):
    """
    Test rate limiting behavior.
    
    Args:
        url: API endpoint URL
        api_key: API key
        requests_per_minute: Expected rate limit
    """
    print(f"\n{'='*80}")
    print(f"RATE LIMIT TEST")
    print(f"{'='*80}")
    print(f"Expected limit: {requests_per_minute}/min")
    print(f"Sending {requests_per_minute + 10} requests rapidly...")
    print(f"{'='*80}\n")
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            make_request(session, url, api_key, "Test query")
            for _ in range(requests_per_minute + 10)
        ]
        
        results = await asyncio.gather(*tasks)
    
    # Analyze
    successful = sum(1 for r in results if r["success"])
    rate_limited = sum(1 for r in results if r["status"] == 429)
    
    print(f"Results:")
    print(f"  Total requests: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Rate limited (429): {rate_limited}")
    print(f"  Other failures: {len(results) - successful - rate_limited}")
    
    if rate_limited > 0:
        print(f"\n✓ Rate limiting working! {rate_limited} requests blocked")
    else:
        print(f"\n⚠ No rate limiting detected")


async def multi_tenant_test(
    url: str,
    tenant_keys: List[str],
):
    """
    Test multi-tenant isolation.
    
    Args:
        url: API endpoint URL
        tenant_keys: List of API keys for different tenants
    """
    print(f"\n{'='*80}")
    print(f"MULTI-TENANT TEST")
    print(f"{'='*80}")
    print(f"Testing {len(tenant_keys)} tenants concurrently...")
    print(f"{'='*80}\n")
    
    async with aiohttp.ClientSession() as session:
        # Send concurrent requests from different tenants
        tasks = []
        for i, api_key in enumerate(tenant_keys):
            for j in range(10):  # 10 requests per tenant
                tasks.append(
                    make_request(
                        session,
                        url,
                        api_key,
                        f"Query {j} from tenant {i}",
                    )
                )
        
        results = await asyncio.gather(*tasks)
    
    # Analyze per tenant
    print(f"Results:")
    for i, api_key in enumerate(tenant_keys):
        tenant_results = results[i*10:(i+1)*10]
        successful = sum(1 for r in tenant_results if r["success"])
        avg_time = statistics.mean(r["duration_ms"] for r in tenant_results if r["success"]) if successful > 0 else 0
        
        print(f"  Tenant {i+1}:")
        print(f"    Successful: {successful}/10")
        print(f"    Avg response time: {avg_time:.2f}ms")


def main():
    """Run load tests."""
    print("\n" + "="*80)
    print("PRODUCTION LOAD TESTING (Phase 4)")
    print("="*80)
    
    # Configuration
    BASE_URL = "http://localhost:8000"
    QUERY_ENDPOINT = f"{BASE_URL}/query"
    
    print("\n⚠️  NOTE: This is a simulation")
    print("   In production, use actual API endpoint and keys")
    
    # Simulated tests
    print("\n1. Standard Load Test")
    print("   Would test 100 requests with concurrency 10")
    
    print("\n2. Rate Limiting Test")
    print("   Would verify rate limits are enforced")
    
    print("\n3. Multi-Tenant Test")
    print("   Would test tenant isolation")
    
    print("\n" + "="*80)
    print("Load Testing Guidelines:")
    print("="*80)
    print("""
1. Start API server:
   python scripts/run_server.py

2. Create test tenants and keys:
   python scripts/tenant_demo.py

3. Run load tests:
   - Uncomment asyncio.run() calls below
   - Update API keys
   - Adjust parameters

4. Monitor:
   - Response times
   - Error rates
   - Rate limit headers
   - System resources

5. Optimize:
   - Adjust rate limits
   - Scale infrastructure
   - Enable caching
   - Add load balancer
    """)
    
    # To actually run tests, uncomment:
    # asyncio.run(load_test(QUERY_ENDPOINT, "YOUR_API_KEY", num_requests=100, concurrency=10))
    # asyncio.run(rate_limit_test(QUERY_ENDPOINT, "YOUR_API_KEY", requests_per_minute=60))
    # asyncio.run(multi_tenant_test(QUERY_ENDPOINT, ["KEY1", "KEY2", "KEY3"]))


if __name__ == "__main__":
    main()
