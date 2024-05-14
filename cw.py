import requests
from bs4 import BeautifulSoup
import os

def save_image(url, directory):
    try:
        image_content = requests.get(url).content
        with open(directory, 'wb') as f:
            f.write(image_content)
    except Exception as e:
        print(f"Failed to save image: {e}")

def download_images(query, num_images=30):
    search_url = f"https://search.naver.com/search.naver?where=image&sm=tab_jum&query={query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    image_tags = soup.find_all('img')

    directory = f"./{query}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    count = 0
    for image_tag in image_tags:
        if count == num_images:
            break
        try:
            image_url = image_tag['src']
            save_image(image_url, f"{directory}/image{count}.jpg")
            count += 1
        except Exception as e:
            print(f"Failed to download image: {e}")

    print(f"Downloaded {count} images of {query}.")

if __name__ == "__main__":
    query = input("검색어를 입력하세요: ")
    download_images(query)
