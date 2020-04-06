# orientation.ch
# The ones to crawl for each vocation are:
# The tree is:
#   div id="content"
#   div class="cont"
#   div class="toggleWrapper"
#   first div with class "toggleBox"
#   div class="boxContent"
# The div contains the skills in <ul>s (every <li> being one task)
# The <h3>s are the umbrella titles but we won't be needing them

# How to get to the list of all vocations:
# https://www.orientation.ch/dyn/show/1922
#   div id="content"
#   div class="cont"
#   div class="toggleWrapper"
#   third div with class "toggleBox"
#   div class="boxContent"
# The div has a list of urls, each being a group of vocations.
# Then for each list page:
#   div id="result-wrapper"
# This contains a list of divs with class="result", each of which then has a div class="title" which contains the <a>.
# Following each of those links leads you to a vocation, whose tasks you can then crawl.

import requests
from bs4 import BeautifulSoup
import pandas as pd
import googletrans
from utilities.common_utils import make_sure_path_exists
from utilities.text_utils import remove_punkt
import pickle
import os
import argparse

BASE_URL = 'https://www.orientation.ch'
STARTING_URL = 'https://www.orientation.ch/dyn/show/1922'
YANDEX_KEY = 'trnsl.1.1.20200401T085601Z.a2a0989bcd9a4117.6be29f3679a7ee3af61f9fbcfa716ce7231a1a2e'
YANDEX_URL = 'https://translate.yandex.net/api/v1.5/tr.json/translate'

def clean_vocation_name(vocation_name):
    vocation_name = remove_punkt(('-'.join(vocation_name.split('-')[:-1]).split('/')[0]).replace('CFC', '')).strip()
    return vocation_name

def get_individual_vocation(url, industry=None):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    vocation_name = soup.head.title.text
    vocation_name = clean_vocation_name(vocation_name)
    skills_list = soup.find('div', {'id': 'content'}).find('div', {'class': 'cont'}).\
                       find('div', {'class': 'toggleWrapper'}).findAll('div', {'class': 'toggleBox'})[0].\
                       find('div', {'class': 'boxContent'}).findAll('li')
    skills_list = [x.text for x in skills_list]
    if industry is not None:
        return pd.DataFrame({'job': [vocation_name]*len(skills_list), 'task': skills_list,
                         'industry': [industry]*len(skills_list)})
    else:
        return pd.DataFrame({'job': [vocation_name] * len(skills_list), 'task': skills_list})


def get_list_of_vocation_urls(starting_page_url, base_url):
    first_page = requests.get(starting_page_url)
    primordial_soup = BeautifulSoup(first_page.text, 'html.parser')
    url_list = primordial_soup.find('div', {'id': 'content'}).find('div', {'class': 'cont'}).\
                                    find('div', {'class': 'toggleWrapper'}).\
                                    findAll('div', {'class': 'toggleBox'})[2].\
                                    find('div', {'class': 'boxContent'}).findAll('a')
    url_list = {a_tag.text: a_tag.get('href') for a_tag in url_list}
    vocation_urls = dict()
    for industry, url in url_list.items():
        response = requests.get(base_url+url)
        soup = BeautifulSoup(response.text, 'html.parser')
        vocations = soup.find('div', {'id': 'result-wrapper'}).findAll('div', {'class': 'result'})
        vocations = [div_tag.find('div', {'class': 'title'}).find('a').get('href') for div_tag in vocations]
        vocation_urls[industry] = vocations
    return vocation_urls

def get_all_vocations(starting_page_url, base_url):
    vocation_urls = get_list_of_vocation_urls(starting_page_url, base_url)
    df_list = list()
    for industry in vocation_urls:
        current_industry_urls = vocation_urls[industry]
        for url in current_industry_urls:
            df_list.append(get_individual_vocation(base_url+url, industry=industry))
    return pd.concat(df_list, axis=0)

def translate_df(df, translation_dir='fr-en'):
    base_url = YANDEX_URL +'?key=' + YANDEX_KEY +'&lang=' + translation_dir + '&text='
    for col_name in df.columns.values:
        df[col_name+'_en'] = df[col_name].apply(lambda x: requests.get(base_url+x).json()['text'][0])
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--crawl', action='store_true')
    parser.add_argument('--translate', action='store_true')
    args = parser.parse_args()
    if args.crawl or not args.translate:
        jobs_df = get_all_vocations(STARTING_URL, BASE_URL)
        print('crawling jobs completed')
        print(jobs_df.head())
        make_sure_path_exists(args.output_dir)
        with open(os.path.join(args.output_dir, 'skills_fr.pkl'), 'wb') as f:
            pickle.dump(jobs_df, f)
    if args.translate:
        if not args.crawl:
            jobs_df = pickle.load(open(os.path.join(args.output_dir, 'skills_fr.pkl'), 'rb'))
        print('starting translation')
        translated_df = translate_df(jobs_df)
        with open(os.path.join(args.output_dir, 'skills_en.pkl'), 'wb') as f:
            pickle.dump(translated_df, f)


if __name__ == '__main__':
    main()

