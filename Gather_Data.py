#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:50:39 2020

@author: wasilaq
"""

import re
from time import sleep
from requests import get
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pymongo import MongoClient

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920x1080")

driver = webdriver.Chrome(options=chrome_options, executable_path='/Applications/chromedriver')
driver.get('https://poets.org/poems?field_poem_themes_tid=1166')

titles = []
authors = []
years = []
poems = []
    
for page in range(1,23): #(1,27)

    # get authors
    page_authors = driver.find_elements_by_css_selector("td[data-label='Author']")
    page_authors = [author.text for author in page_authors]
    authors.extend(page_authors)
    
    # get years
    page_years = driver.find_elements_by_css_selector("td[data-label='Year']")
    page_years = [year.text for year in page_years]
    years.extend(page_years)
    
    for poem in range(1,21):
        
        # get title
        title_path = '//*[@id="poems"]/tbody/tr[{}]/td[1]/div/a'.format(poem)
        wait = WebDriverWait(driver, 10)
        title_element = wait.until(EC.element_to_be_clickable((By.XPATH, title_path)))
        
        link = title_element.get_attribute('href')
        title_element = wait.until(EC.element_to_be_clickable((By.XPATH, title_path))) # staleelementexception
        title = title_element.text
        titles.append(title)
        
        sleep(2)
        
        # get poem body
        poem_page = get(link).text
        poem_soup = BeautifulSoup(poem_page, 'lxml')
        poem_loc = poem_soup.find(class_='card card--main').find('div', class_='card-body').find('div', class_=re.compile('poem__body px-md-4 font-serif'))
        lines = poem_loc.findAll(text=True)
        poem = '+'.join([line for line in lines])
        poems.append(poem)
        
        sleep(2)
    

    print('Done collecting poems for page ' + str(page))
    next_page = driver.find_element_by_xpath('//*[@id="__layout"]/div/div[4]/div[3]/div/ul/li[7]/a/span/span')
    next_page.click()
    sleep(10)
    

# years and authors lists don't include first 20 poems, didn't get first page    
# store data in mongodb
db = MongoClient().poetry
collection = db.love_poems

for title, poem, year, author in list(zip(titles[20:], poems[20:], years, authors)):
    document = {'title': title, 'poem': poem, 'year': year, 'author': author}
    collection.insert_one(document)