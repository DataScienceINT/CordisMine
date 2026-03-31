import prodigy
import csv
import random
from prodigy.components.stream import Stream

@prodigy.recipe(
	"custom-textcat",
	dataset=("The dataset to use", "positional", None, str))

def custom_textcat(dataset, source):
    blocks = [
        {"view_id": "choice"}
    ]
    options = [
        {"id": 1, "text": "😺 Relevant"},
        {"id": 0, "text": "😾 Not relevant"}
    ]
    def custom_csv_loader(source, encoding='utf-8-sig'): 
        with open(source, mode="r", encoding="utf-8-sig") as csvfile: 
            reader = csv.DictReader(csvfile)
            rows = list(reader)  
            random.shuffle(rows)  
            for row in rows: 
                training = "TRAINING SET\n\\n" if row.get('Include')=="relevant" else ""
                text = training + row.get('title') + '\n\n' + row.get('objective')
                id = row.get('id')
                acronym = row.get('acronym')
                yield {"text": text, "options": options, "meta": {"id":id, "acronym": acronym}}

    stream_as_generator = custom_csv_loader(source)
    stream = Stream.from_iterable(stream_as_generator) 

    return {
        "dataset": dataset,         
        "view_id": "blocks",         
        "stream": stream,            
        "config": {
            "labels": [""],
            "blocks": blocks
        }
    }
