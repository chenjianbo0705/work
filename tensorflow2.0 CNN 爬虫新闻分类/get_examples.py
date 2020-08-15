import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_nba():
    r = requests.get('http://sports.sina.com.cn/nba/5.shtml')
    r.encoding = 'utf-8'
    soup = BeautifulSoup(r.text, 'html.parser')
    news_dict = {'title': [], 'href': [], 'time': [], 'acticle': []}
    for q in range(11, 20):
        print('S_Cont_{page}'.format(page=q))
        new = soup.find(id='S_Cont_{page}'.format(page=q))
        ss = new.find_all('a')
        sd = new.find_all('span')
        for i in tqdm(ss):
            href = i['href']
            title = i.text
            news_dict['title'].append(title)
            news_dict['href'].append(href)
        for w in sd:
            data = w.text
            news_dict['time'].append(data)
    for e in tqdm(news_dict['href']):
        try:
            re = requests.get(e)
            re.encoding = 'utf-8'
            sou = BeautifulSoup(re.text, 'html.parser')
            new = sou.find('div', 'article')
            st1 = new.text.replace(u'\n\u3000\u3000', u' ').replace(u'\n\n\n\n', u'').replace(u'\n\n', u'')
        except:
            print('地址为{}新闻抓取失败'.format(e))
            continue
        news_dict['acticle'].append(st1)
        df = pd.DataFrame.from_dict(news_dict, orient='index')
        df.to_excel('NBA2.xlsx')
def get_keji():
    try:
        r = requests.get('https://tech.sina.com.cn')
        r.encoding = r.apparent_encoding
    except:
        print('网页源代码获取失败')
    news_dict = {'title': [], 'href': [], 'time': [], 'acticle': []}
    soup = BeautifulSoup(r.text, 'html.parser')
    new = soup.find('ul', 'seo_data_list')
    srt = new.find_all('li')
    from tqdm import tqdm
    for i in tqdm(srt):
        href = i('a')[0]['href']
        title = i.text
        news_dict['title'].append(title)
        news_dict['href'].append(href)
    for i in tqdm(news_dict['href']):
        try:
            res = requests.get(i)
            res.encoding = 'utf-8'
            soup = BeautifulSoup(res.text, 'html.parser')
            new = soup.find('div', 'article').text
            # new  = re.sub(reg,'',new)
            data = soup.find('span', 'date').text
        except:
            print('地址为{}新闻抓取失败'.format(i))
            continue
        news_dict['acticle'].append(new)
        news_dict['time'].append(data)
        df = pd.DataFrame.from_dict(news_dict, orient='index')
        df.to_excel('./data/keji2.xlsx')


if __name__ == '__main__':
    get_nba()
    get_keji()