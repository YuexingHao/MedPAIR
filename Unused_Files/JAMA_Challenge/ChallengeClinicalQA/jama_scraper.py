import pandas as pd
import numpy as np
import time
import random
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup

def setup_driver():
    """Set up and return a configured Chrome WebDriver"""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    
    driver = webdriver.Chrome(options=options)
    return driver

def extract_paragraphs(soup):
    casepara = []
    case = 0
    discusiion = 0
    discussionpara = []
    paragraphs = []
    diagnosis = ""
    fetchtrue = False
    founddiagnosis = False
    fetchoptiontrue = False
    chooseoption = ""
    
    if soup:
        article_para = soup.find('div', class_='article-full-text')
        stop_condition = "Article Information"
        for paragraph in article_para.find_all(['div', 'p']):
            if paragraph.name == 'div':
                if stop_condition in paragraph.get_text():
                    break
            else:
                if fetchoptiontrue:
                    chooseoption = paragraph.get_text()
                    fetchoptiontrue = False
                elif fetchtrue:
                    diagnosis = paragraph.get_text()
                    fetchtrue = False
                    founddiagnosis = True
                elif paragraph.get_text() == "Diagnosis":
                    fetchtrue = True
                elif paragraph.get_text() in ["What to Do Next", "What To Do Next", "Answer"]:
                    if paragraph.get_text() in ["Answer", "What To Do Next"]:
                        print(paragraph.get_text())
                    fetchoptiontrue = True
                    founddiagnosis = False
                else:
                    if paragraph.get_text() == "Case":
                        case = 1
                    if paragraph.get_text() == "Discussion":
                        case = 0
                        discusiion = 1
                    if np.char.count(paragraph.get_text(), ' ') + 1 < 8:
                        continue
                    if case == 1:
                        casepara.append(paragraph.get_text())
                    if discusiion == 1:
                        discussionpara.append(paragraph.get_text())
                    paragraphs.append(paragraph.get_text())

        if founddiagnosis and not fetchoptiontrue:
            chooseoption = diagnosis
            
    return paragraphs, diagnosis, chooseoption, casepara, discussionpara

def hasImage(soup):
    article_para = soup.find('div', class_='article-full-text')
    if article_para:
        image_div = article_para.find('div', class_='figure-table-image')
        if image_div and image_div.find('img'):
            return True
    return False

def tellfield(soup):
    article_para = soup.find('div', class_='meta-article-type thm-col')
    super_class = soup.find('div', class_='meta-super-class')
    if super_class:
        return article_para.get_text(), super_class.get_text()
    return article_para.get_text(), None

def extractMCQ(soup):
    ques = None
    ans = None
    if soup:
        div_element = soup.find('div', class_='box-section online-quiz clip-last-child thm-bg')
        if div_element is None:
            return None, ques, ans
        
        question_element = div_element.find('h4', class_='box-section--title')
        p_elements = div_element.find_all('p', class_='para')
        
        question = question_element.text
        answers = [p.text for p in p_elements]
        
        whetherTable = 1
        ques = question
        return whetherTable, ques, answers

def get_page_content(driver, url, max_retries=2):
    """Get page content with retries using Selenium"""
    for attempt in range(max_retries):
        try:
            driver.get(url)
            # Wait for the article content to load
            # WebDriverWait(driver, 10).until(
            #     EC.presence_of_element_located((By.CLASS_NAME, "article-content"))
            # )
            time.sleep(10)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            print("DEBUG: Response Text Begins ----------")
            print(soup.text[:1000])  # limit output
            print("DEBUG: Response Text Ends ----------")

            return soup
        except TimeoutException:
            if attempt == max_retries - 1:
                print(f"Failed to load {url} after {max_retries} attempts")
                return None
            print(f"Timeout on attempt {attempt + 1}, retrying...")
            time.sleep(random.uniform(2, 4))
    return None

if __name__ == '__main__':
    url_df = pd.read_json('jama_links.json', orient='records')
    url_df = url_df.drop(columns='id')
    dff = []
    
    driver = setup_driver()
    try:
        cnt = 0
        print("Start Scraping...")
        
        for index, row in url_df.iterrows():
            url = row['link']
            soup = get_page_content(driver, url)            
            
            if soup is None:
                print("Failed to load page, skipping...")
                continue
            
            whethermcq, mcqquestion, answers = extractMCQ(soup)
            if whethermcq is None:
                import IPython; IPython.embed()
                print("No MCQ found....trying again ")
                time.sleep(random.uniform(1, 2))
                soup = get_page_content(driver, url)
                
                if soup is None or extractMCQ(soup)[0] is None:
                    print("Please check your license to ensure you have access to JAMA website.")
                    continue
                    
                whethermcq, mcqquestion, answers = extractMCQ(soup)

            paragraphs, diagnosis, chooseoption, casepara, discussionpara = extract_paragraphs(soup)
            checkImage = hasImage(soup)
            HasImage = "Yes" if checkImage else "No"
            
            articleType, superclass = tellfield(soup)
            
            combineCasepara = "".join(casepara)
            combinediscussionpara = "".join(discussionpara)
            question = combineCasepara + ' ' + mcqquestion
            
            # We directly copy the answer from jama_links.json to make sure they are correct
            dff.append([
                url, question, answers[0], answers[1], answers[2], answers[3],
                diagnosis, row['answer_idx'], row['answer'], combinediscussionpara, articleType
            ])
            
            cnt += 1
            if cnt % 10 == 0:
                print(f"{cnt} Links are Successfully Fetched")
                
            # Be nice to the server
            time.sleep(random.uniform(1, 2))

            print(dff[-1])
            
    finally:
        driver.quit()
        
    df = pd.DataFrame(dff, columns=[
        'link', 'question', 'opa', 'opb', 'opc', 'opd',
        'diagnosis', 'answer_idx', 'answer', 'explanation', 'field'
    ])
    
    print("Scraping Finished")
    
    # Save to CSV
    df.to_csv("jama_raw.csv", index=False)
    
    # Save to JSON
    df.index.name = 'id'
    df = df.reset_index()
    json_dict = df.to_dict(orient='records')
    with open('jama_raw.json', 'w') as f:
        json.dump(json_dict, f, indent=4)
        
    print("Files Saved")
