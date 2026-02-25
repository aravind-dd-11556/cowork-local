"""Allow running as: python -m claude_web_tools"""
import asyncio
from .main import main

asyncio.run(main())
