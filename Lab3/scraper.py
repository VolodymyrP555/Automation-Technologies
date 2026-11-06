
import json
import html
import re
import pandas as pd

from playwright.sync_api import sync_playwright

URL = "https://webscraper.io/test-sites/e-commerce/ajax/computers/laptops"
OUTPUT = "laptops_catalog.csv"


def get_processor(description):
    processors = [
        "Intel Core i9", "Intel Core i7", "Intel Core i5",
        "Intel Core i3", "Core i7", "Core i5", "Core i3",
        "Celeron", "Pentium", "Ryzen 9", "Ryzen 7",
        "Ryzen 5", "Ryzen 3", "AMD A12", "AMD A10",
        "AMD A9", "AMD A8", "AMD A6", "AMD E2"
    ]

    for processor in processors:
        if processor.lower() in description.lower():
            return processor

    return "Не вказано"


def main():
    products = []
    responses = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        def capture_response(response):
            if (
                "/test-sites/e-commerce/ajax/computers/laptops"
                in response.url
                and response.request.resource_type == "document"
            ):
                responses.append(response)
                print(
                    f"[NETWORK] HTTP {response.status} "
                    f"| {response.request.resource_type}"
                )

        page.on("response", capture_response)

        def block_resources(route):
            if route.request.resource_type in ["image", "media", "font"]:
                route.abort()
            else:
                route.continue_()

        page.route("**/*", block_resources)

        try:
            print("[INFO] Відкриваємо каталог...")

            response = page.goto(
                URL,
                wait_until="domcontentloaded",
                timeout=30000
            )

            if response is None or response.status != 200:
                raise RuntimeError("Каталог недоступний")

            catalog = page.locator("[data-items]").first
            catalog.wait_for(timeout=15000)

            raw_data = catalog.get_attribute("data-items")
            items = json.loads(raw_data)

            print(f"[INFO] Отримано {len(items)} товарів")

            original_url = page.url

            for page_number in range(2, 6):
                button = page.locator(
                    f"button[data-id='{page_number}']"
                )

                button.click()

                page.wait_for_function(
                    """number => {
                        const button = document.querySelector(
                            `button[data-id="${number}"]`
                        );
                        return button &&
                            button.classList.contains("active");
                    }""",
                    arg=page_number,
                    timeout=10000
                )

                print(
                    f"[PAGINATION] Перейшли на сторінку "
                    f"{page_number}"
                )

            if page.url == original_url:
                print("[OK] URL залишився незмінним")

            for item in items:
                description = item.get("description", "")

                products.append({
                    "id": item.get("id"),
                    "model": item.get("title"),
                    "price": item.get("price"),
                    "processor": get_processor(description),
                    "description": description,
                    "availability": "Не вказано"
                })

            df = pd.DataFrame(products)
            df = df.drop_duplicates(subset=["id"])

            df.to_csv(
                OUTPUT,
                index=False,
                encoding="utf-8-sig"
            )

            print(f"\n[OK] Збережено {len(df)} ноутбуків")
            print(f"[OK] Файл: {OUTPUT}")

            print("\nПерші 10 записів:")
            print(df.head(10).to_string(index=False))

            print(
                f"\n[NETWORK] Перехоплено відповідей "
                f"каталогу: {len(responses)}"
            )

        except Exception as error:
            print(f"[ERROR] {error}")

        finally:
            input("\nНатисни Enter, щоб закрити браузер...")
            browser.close()


if __name__ == "__main__":
    main()

