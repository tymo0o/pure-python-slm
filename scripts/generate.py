import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model.language_model import SimpleLM
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate text using the trained model')
    parser.add_argument('--model', type=str, default='output/model.json',
                      help='Path to the trained model file')
    parser.add_argument('--seed', type=str, default='Once upon a',
                      help='Seed text to start generation')
    parser.add_argument('--length', type=int, default=50,
                      help='Number of words to generate')
    
    args = parser.parse_args()
    
    # Load model
    model = SimpleLM.load(args.model)
    
    # Generate text
    generated = model.generate_text(args.seed, args.length)
    print(f"\nGenerated text:\n{generated}")

if __name__ == "__main__":
    main()
