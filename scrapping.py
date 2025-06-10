from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
from datetime import datetime
import time

start_urls = [
    "https://tds.s-anand.net/#/2025-01/", "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34"
]

USERNAME = "24ds1000012@ds.study.iitm.ac.in"
PASSWORD = "1234567@com"

# Define helper functions
def is_valid_tds_link(url: str):
    for month in ['2025-01', '2025-02', '2025-03', '2025-04']:
        if f"/#/{month}/" in url:
            if month == '2025-04':
                try:
                    day_part = int(url.split("/#/2025-04/")[1][:2])
                    return day_part <= 14
                except:
                    return False
            return True
    return False

with sync_playwright() as p:
    DEBUG_MODE = False
    browser = p.chromium.launch(headless=not DEBUG_MODE)
    context = browser.new_context()
    page = context.new_page()

    for start_url in start_urls:
        page.goto(start_url)
        page.wait_for_load_state("networkidle")

        login_button_keywords = [
            "login", "sign in", "log in", "authenticate", "access account",
            "Log In", "Sign In", "Login", "Log In", "Sign in", "Access Account"
        ]
        login_input_fields = ["login-account-name", "login-account-password", "username", "password", "email"]
        login_required = False

        print("\n🧾 Checking login requirements...")

        buttons = page.query_selector_all("button")
        for button in buttons:
            label = (button.inner_text() or button.get_attribute("aria-label") or "").strip().lower()
            if any(keyword.lower() in label for keyword in login_button_keywords):
                login_required = True
                break

        inputs = page.query_selector_all("input")
        for input_tag in inputs:
            name = input_tag.get_attribute("name")
            if name and name.lower() in login_input_fields:
                login_required = True
                break

        if login_required:
            print("🔐 Login required. Attempting login...")
            login_links = page.query_selector_all("a")
            for link in login_links:
                text = (link.inner_text() or "").strip().lower()
                if any(keyword.lower() in text for keyword in login_button_keywords):
                    print(f"🔘 Clicking login link: {text}")
                    link.click()
                    page.wait_for_load_state("networkidle")
                    time.sleep(1)
                    break

            try:
                page.wait_for_selector('#login-account-name', timeout=10000)
                page.fill('#login-account-name', USERNAME)
                page.fill('#login-account-password', PASSWORD)
                page.click("#login-button")
                page.wait_for_load_state("networkidle")
                time.sleep(2)
                page.wait_for_selector('a[href*="/u/"]', timeout=5000)
                print("✅ Login successful.")
            except:
                print("❌ Login failed. Skipping this URL.")
                continue

        # Extract links
        print("🔗 Collecting page links...")
        links = page.query_selector_all("a")
        hrefs = []
        for link in links:
            href = link.get_attribute("href")
            if href and not href.startswith("mailto:"):
                full_url = urljoin(start_url, href)
                hrefs.append(full_url)

        print(f"🔍 Found {len(hrefs)} valid links.")

        # Extract content from each link
        contents = []
        for href in hrefs:
            try:
                print(f"🌐 Visiting: {href}")
                page.goto(href)
                page.wait_for_load_state("networkidle")
                body = page.query_selector("body")
                content = body.inner_text() if body else ""
                contents.append(content)
                print(f"✅ Extracted from: {href}")
            except Exception as e:
                print(f"⚠️ Failed to extract from {href}: {e}")
                continue

        with open("extracted_contents_2.doc", "a", encoding="utf-8") as f:
            for content in contents:
                f.write(content + "\n\n")

        print(f"📁 Contents saved for URL: {start_url}")

    context.close()
    browser.close()
    print("✅ All done.")