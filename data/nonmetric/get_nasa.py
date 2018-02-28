import os
import random
import urllib2
from bs4 import BeautifulSoup as bsoup

# download html with links to all articles
link = "https://www.nasa.gov/audience/formedia/archives/MP_Archive_02.html"
html = urllib2.urlopen(urllib2.Request(link, headers={'User-Agent': 'Magic Browser%i'%random.randint(0,100)})).read()
# make a beautiful soup object out of the html
soup = bsoup(html)
# extract the links to all articles
txt_urls = soup.find_all('a', {'class':'featureLnk'}, href=True)
# save all articles (beware: folder has to exist!)
for u in txt_urls:
    with open(os.path.join('nasa', os.path.basename(u['href'])), 'w') as f:
        f.write(urllib2.urlopen(urllib2.Request("https://www.nasa.gov" + u['href'], headers={'User-Agent': 'Magic Browser%i'%random.randint(0,100)})).read())
