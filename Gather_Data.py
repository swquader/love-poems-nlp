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
soup = BeautifulSoup(driver.page_source, 'html.parser')

# get title, author, and year
links = []
titles = []
authors = []
years = []

for page in range(1,29): #orinally (1,10)
    
    for poem in range(1,21):
        
        title_path = '//*[@id="poems"]/tbody/tr[{}]/td[1]/div/a'.format(poem)
        wait = WebDriverWait(driver, 10)
        title_element = wait.until(EC.element_to_be_clickable((By.XPATH, title_path)))
        title = title_element.text
        titles.append(title)
        
        title_element = wait.until(EC.element_to_be_clickable((By.XPATH, title_path))) # called again to fix StaleElementReferenceException
        links.append(title_element.get_attribute('href'))
        
        author_path = '//*[@id="poems"]/tbody/tr[{}]/td[2]/div'.format(poem)
        author_element = driver.find_element_by_xpath(author_path)
        author = author_element.text
        authors.append(author)
        
        year_path = '//*[@id="poems"]/tbody/tr[{}]/td[3]/div'.format(poem)
        year_element = driver.find_element_by_xpath(year_path)
        year = year_element.text
        years.append(year)
        
        sleep(1)
    
    sleep(10)
    # go to next page to grab more poems
    next_page = driver.find_element_by_xpath('//*[@id="__layout"]/div/div[4]/div[3]/div/ul/li[3]/a')
    next_page.click()
    
# get poem body
poems = []

for link in links:
    poem_page = get(link).text
    poem_soup = BeautifulSoup(poem_page, 'lxml')
    poem_loc = poem_soup.find(class_='card card--main').find('div', class_='card-body').find('div', class_=re.compile('poem__body px-md-4 font-serif'))
    # lines = poem_loc.find('p').find_all('span', class_='long-line')
    lines = poem_loc.findAll(text=True)
    # lines = [line.strip() for line in grab_lines]
    poem = '+'.join([line for line in lines])
    
    poems.append(poem)
    sleep(2)
    
# store data in mongodb
db = MongoClient().poetry
collection = db.love_poems

for item in range(560):
    document = {'title': titles[item], 'poem': poems[item], 'year': years[item], 'author': authors[item], 'link': links[item]}
    collection.insert_one(document)