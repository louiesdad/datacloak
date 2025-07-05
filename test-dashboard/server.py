#!/usr/bin/env python3
"""
DataCloak Test Dashboard Server
Serves the test dashboard and provides API endpoints for running tests
"""

import os
import json
import subprocess
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from aiohttp import web
import aiohttp_cors

@dataclass
class TestResult:
    name: str
    status: str
    duration: str
    total: int
    passed: int
    failed: int
    output: str
    timestamp: str

class TestRunner:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_results: Dict[str, TestResult] = {}
        self.is_running = False
        
    async def run_test(self, test_name: str, test_type: str) -> TestResult:
        """Run a single test and capture results"""
        start_time = time.time()
        
        cmd = [
            "cargo", "test", "--test", f"{test_name}_tests", 
            "--", "--nocapture", "--test-threads=4"
        ]
        
        try:
            # Run test
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_root / "datacloak-core",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            stdout, _ = await process.communicate()
            output = stdout.decode('utf-8')
            
            # Parse results
            total = output.count("test ") - output.count("test result:")
            passed = output.count(" ... ok")
            failed = output.count(" ... FAILED")
            
            status = "passed" if process.returncode == 0 else "failed"
            duration = f"{time.time() - start_time:.2f}s"
            
            result = TestResult(
                name=test_name,
                status=status,
                duration=duration,
                total=total,
                passed=passed,
                failed=failed,
                output=output[-2000:],  # Last 2000 chars
                timestamp=datetime.now().isoformat()
            )
            
            self.test_results[test_name] = result
            return result
            
        except Exception as e:
            return TestResult(
                name=test_name,
                status="error",
                duration="0s",
                total=0,
                passed=0,
                failed=0,
                output=str(e),
                timestamp=datetime.now().isoformat()
            )
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all tests"""
        if self.is_running:
            return []
        
        self.is_running = True
        
        unit_tests = ["adaptive_sampling", "bounded_cache", "bounded_obfuscator", "thread_config"]
        integration_tests = ["enhanced_integration", "streaming_detection"]
        performance_tests = ["performance_optimization", "performance_regression"]
        
        all_tests = [
            (test, "unit") for test in unit_tests
        ] + [
            (test, "integration") for test in integration_tests
        ] + [
            (test, "performance") for test in performance_tests
        ]
        
        results = []
        for test_name, test_type in all_tests:
            result = await self.run_test(test_name, test_type)
            results.append(result)
            
        self.is_running = False
        return results

# Web server setup
async def get_test_results(request):
    """Get all test results"""
    runner = request.app['test_runner']
    results = {
        'results': [asdict(r) for r in runner.test_results.values()],
        'is_running': runner.is_running,
        'timestamp': datetime.now().isoformat()
    }
    return web.json_response(results)

async def run_tests(request):
    """Run all tests"""
    runner = request.app['test_runner']
    
    if runner.is_running:
        return web.json_response({'error': 'Tests already running'}, status=400)
    
    # Run tests in background
    asyncio.create_task(runner.run_all_tests())
    
    return web.json_response({'status': 'started'})

async def run_single_test(request):
    """Run a single test"""
    runner = request.app['test_runner']
    test_name = request.match_info['test_name']
    
    result = await runner.run_test(test_name, "unit")
    
    return web.json_response(asdict(result))

async def get_coverage(request):
    """Get code coverage report"""
    # Run coverage command
    cmd = ["cargo", "tarpaulin", "--print-summary"]
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=request.app['project_root'] / "datacloak-core",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return web.json_response({
            'coverage': stdout.decode('utf-8'),
            'error': stderr.decode('utf-8') if stderr else None
        })
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def websocket_handler(request):
    """WebSocket for real-time updates"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app['websockets'].add(ws)
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                if msg.data == 'close':
                    await ws.close()
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f'WebSocket error: {ws.exception()}')
    finally:
        request.app['websockets'].discard(ws)
        
    return ws

async def broadcast_update(app, data):
    """Broadcast update to all WebSocket clients"""
    if app['websockets']:
        await asyncio.gather(
            *[ws.send_json(data) for ws in app['websockets']],
            return_exceptions=True
        )

def create_app(project_root: Path):
    """Create the web application"""
    app = web.Application()
    
    # Initialize test runner
    app['test_runner'] = TestRunner(project_root)
    app['project_root'] = project_root
    app['websockets'] = set()
    
    # Setup CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    
    # Routes
    app.router.add_get('/api/results', get_test_results)
    app.router.add_post('/api/run', run_tests)
    app.router.add_post('/api/run/{test_name}', run_single_test)
    app.router.add_get('/api/coverage', get_coverage)
    app.router.add_get('/ws', websocket_handler)
    
    # Serve static files with cache control
    async def serve_index(request):
        response = web.FileResponse(project_root / 'test-dashboard' / 'index.html')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
    app.router.add_get('/', serve_index)
    app.router.add_static('/', path=project_root / 'test-dashboard', name='static', show_index=True)
    
    # Configure CORS on all routes
    for route in list(app.router.routes()):
        if not isinstance(route.resource, web.StaticResource):
            cors.add(route)
    
    return app

async def main():
    """Main entry point"""
    project_root = Path(__file__).parent.parent
    app = create_app(project_root)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, 'localhost', 9090)
    await site.start()
    
    print("Dashboard server running at http://localhost:9090")
    print("Press Ctrl+C to stop")
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())