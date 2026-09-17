import asyncio
from crawl4ai import AsyncWebCrawler

async def save_markdown(url: str, out_path: str = "page.md"):
    async with AsyncWebCrawler() as crawler:           # 启动无头浏览器
        result = await crawler.arun(url)               # 抓取 + HTML→Markdown
        print(result)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result.markdown)                   # 写入纯 Markdown
    print(f"✓ 已保存到 {out_path}")

if __name__ == "__main__":
    target = "https://baike.baidu.com/starmap/view?fromModule=starMap_recommend&lemmaId=938188&lemmaTitle=%E6%B9%96%E5%8C%97%E7%9C%81%E5%8D%9A%E7%89%A9%E9%A6%86&nodeId=cf1d6ae7cd301bd970d36dd7&starMapFrom=lemma_starMap"
    savep = r"D:\ddesktop\agentdemos\codespace\hubeipm\text1.md"
    asyncio.run(save_markdown(target, savep))
