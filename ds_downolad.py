from datasets import load_dataset
import csv
import os
from ds_preprocess import clean_code 

languages = open('selected_languages.txt').read().splitlines()

with open('dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(['language', 'content'])  # header

    for language in languages:
        print(f"[{language}] Pobieranie...", flush=True)
        try:
            ds = load_dataset(
                "bigcode/the-stack",
                data_dir=f"data/{language}",
                split="train",
                streaming=True
            )

            count = 0
            for i, example in enumerate(ds):
                if i >= 5000:
                    break
                content = example['content'].replace('\r\n', '\n')
                content = clean_code(content, language)
                writer.writerow([language, content])
                count += 1

            f.flush()
            print(f"[{language}] OK — {count} przykładów", flush=True)

        except Exception as e:
            print(f"[{language}] BŁĄD: {e}", flush=True)

os._exit(0)