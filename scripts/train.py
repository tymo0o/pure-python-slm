import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model.language_model import SimpleLM
from model.utils import download_tinystories, load_dataset
import os

def main():
    # Download dataset if it doesn't exist
    data_dir = Path(__file__).parent.parent / 'data'
    data_file = data_dir / 'sample_data.txt'
    
    if not data_file.exists():
        stories = download_tinystories(save_dir=data_dir)
    else:
        stories = load_dataset(data_file)
    
    # Initialize and train model
    print("Training model...")
    model = SimpleLM(vocab_size=5000, embedding_dim=64, context_size=3)
    model.train(stories, learning_rate=0.01, epochs=5)
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Save the trained model
    model_path = output_dir / 'model.json'
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Generate sample text
    seed_text = "Once upon a"
    generated = model.generate_text(seed_text, length=50)
    print(f"\nSample generated text:\n{generated}")

if __name__ == "__main__":
    main()
