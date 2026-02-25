import asyncio
from crawl4ai import *

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://docs.catalyst.zoho.com/en/cli/v1/cli-command-reference/",
        )
        internal_links = result.links.get("internal",[])
        hrefs = [link.get("href") for link in internal_links if "href" in link]
        print (hrefs)

if __name__ == "__main__":
    asyncio.run(main())