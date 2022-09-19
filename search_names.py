from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
from names_from_folder import DirNames

dir_names = DirNames('../lfw/')
dir_names.search()
people = dir_names.all_names

# df = pd.DataFrame(columns=['Search person', 'Found person', 'Description'])
df = pd.read_csv('./search_output.csv')

df_names = list(df['Search person'])

opt = ChromeOptions()
opt.add_argument('--headless')
wd = webdriver.Chrome(ChromeDriverManager().install(), options=opt)

wd.get('https://en.wikipedia.org/wiki/Main_Page')

for search_person in people:
    if search_person in df_names:
        continue

    input_field = wd.find_element(By.CLASS_NAME, "vector-search-box-input")
    input_field.clear()
    input_field.send_keys(search_person)

    time.sleep(1)
    suggestion_class = wd.find_element(By.CLASS_NAME, 'suggestions')
    suggestion_res = suggestion_class.find_element(By.CLASS_NAME, 'suggestions-results')

    try:
        suggestion_res_a = suggestion_res.find_element(By.TAG_NAME, 'a')
        suggestion_res_a.click()
        time.sleep(1)
        found_person = wd.find_element(By.ID, 'firstHeading').text
        page_text = wd.find_element(By.ID, "mw-content-text")
        paragraphs = page_text.find_elements(By.TAG_NAME, 'p')
        chosen_par = paragraphs[0].text + '\n' + paragraphs[1].text
        description = chosen_par
        print(chosen_par)
    except NoSuchElementException:
        found_person = 'None'
        description = 'None'
        print('Persona non famosa')
    except IndexError:
        chosen_par = 'ERROR in reading'
    except:
        break

    df = df.append({'Search person': search_person.lower(), 'Found person': found_person.lower(), 'Description': description}, ignore_index=True)
df.to_csv('search_output2.csv')

print()
wd.close()

# input_field.send_keys(Keys.ENTER)

# title = wd.find_element(By.ID, "firstHeading")
# print(title.text)

# if title.text.lower() == 'risultati della ricerca':
#     ul = wd.find_element(By.TAG_NAME, 'ul')
#     print()


