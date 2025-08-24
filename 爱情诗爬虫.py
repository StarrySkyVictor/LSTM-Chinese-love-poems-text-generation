from bs4 import BeautifulSoup
import requests
import re

url = "https://www.gushiwen.cn/gushi/aiqing.aspx"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0 Safari/537.36"
}
# 发送请求
response = requests.get(url, headers=headers)
html = response.text

soup = BeautifulSoup(html, "html.parser")

# 目录页：找到爱情诗分类的 div
div = soup.find("div", class_="typecont")

# 提取所有 a 标签的 href
hrefs = [a["href"] for a in div.find_all("a", href=True)]

# 打开文件准备写入
with open("爱情诗.txt", "w", encoding="utf-8") as f:
    for href in hrefs:
        poem_url = "https://www.gushiwen.cn" + href
        resp = requests.get(poem_url, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")

        contson = soup.find("div", class_="contson")
        text = contson.get_text(separator="\n", strip=True) if contson else ""
        poem = re.sub(r'[^\u4e00-\u9fa5，。！？]', '', text)  # 保留常用标点和中文
        f.write(text)

print("✅ 已保存到 爱情诗.txt")
