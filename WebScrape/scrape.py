import requests
from bs4 import BeautifulSoup
from bs4.element import Comment


# Define a function to check if the element is visible text content.
def tag_visible(element):
    # Exclude elements within 'style', 'script', 'head', or 'meta' tags.
    if element.parent.name in ['style', 'script', 'head', 'meta']:
        return False
    # Exclude comment elements.
    if isinstance(element, Comment):
        return False
    # If none of the above, the element is visible.
    return True


# Define a function to extract visible text from the HTML content.
def text_from_html(soup):
    # Find all string elements in the HTML document.
    texts = soup.findAll(string=True)
    # Use the 'tag_visible' function to filter out non-visible texts.
    visible_texts = filter(tag_visible, texts)
    # Join the visible texts into a single string separated by spaces,
    # stripping any leading/trailing whitespace from each text.
    return u" ".join(t.strip() for t in visible_texts)


# The target URL to scrape.
url = ("https://www.vg.no/rampelys/i/yE1VE2/else-kaass-furuseth-gifter-seg-i-oslo-spektrum-ingen-vet-hvem-den"
       "-utkaarede-er")

# Make an HTTP GET request to the URL.
response = requests.get(url)

# Check if the request was successful.
if response.ok:
    # Parse the HTML content of the page using BeautifulSoup.
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract visible text from the parsed HTML.
    text = text_from_html(soup)
    # Print the extracted text.
    print(text)

    # Open a file for writing and save the extracted text.
    with open("output.txt", "w") as file:
        file.write(str(text))
