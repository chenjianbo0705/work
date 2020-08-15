import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import  numpy as np

def get_poems():
    r = requests.get('http://www.gushicimingju.com/gushi/qiyanjueju/')
    r.encoding = 'utf-8'
    soup = BeautifulSoup(r.text, 'html.parser')
    news_dict = {'content': []}
    new = soup.find_all('span', 'content')
    for i in range(len(new)):
        content = new[i].text
        news_dict['content'].append(content)
    for page in range(2, 22):
        print(page)
        r = requests.get('http://www.gushicimingju.com/gushi/qiyanjueju/page{page}/'.format(page=page))
        print('http://www.gushicimingju.com/gushi/qiyanjueju/page{page}/'.format(page=page))
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'html.parser')
        new = soup.find_all('span', 'content')
        for i in range(len(new)):
            content = new[i].text
            # if len(content)==64:
            # print(content)
            news_dict['content'].append(content)
    df = pd.DataFrame.from_dict(news_dict, orient='index')
    df.transpose().head(122)
    df.to_excel('poems.xlsx')
if __name__ == '__main__':
    get_poems()