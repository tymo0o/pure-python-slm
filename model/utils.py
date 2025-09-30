import requests
import gzip
import json
from pathlib import Path

def download_tinystories(save_dir='data', num_stories=1000):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data/TinyStories_00000.txt.gz"
    response = requests.get(url, stream=True)
    
    stories = []
    output_file = save_dir / 'sample_data.txt'
    
    print(f"Downloading and processing {num_stories} stories...")
    with gzip.open(response.raw) as f:
        for i, line in enumerate(f):
            if i >= num_stories:
                break
            story = json.loads(line)['text']
            stories.append(story)
    
    # Save stories to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for story in stories:
            f.write(story + '\n\n')
    
    print(f"Saved {len(stories)} stories to {output_file}")
    return stories

def load_dataset(filepath):
    """Load dataset from a text file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]
