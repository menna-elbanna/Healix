from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import requests
import os

options = Options()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)

url = "https://www.magnific.com/search?query=cardboard+box+waste"
driver.get(url)

time.sleep(5)

last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

images = driver.find_elements(By.TAG_NAME, "img")

img_links = set()

for img in images:
    src = img.get_attribute("src") or img.get_attribute("data-src")
    if src and "http" in src:
        img_links.add(src)

driver.quit()

folder = "images"
os.makedirs(folder, exist_ok=True)

for i, link in enumerate(img_links):
    try:
        response = requests.get(link, timeout=10)
        if response.status_code == 200:
            with open(f"{folder}/img_{i}.jpg", "wb") as f:
                f.write(response.content)
    except:
        pass