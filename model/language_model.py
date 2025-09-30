import numpy as np
from collections import defaultdict, Counter
import json

class SimpleLM:
    def __init__(self, vocab_size=5000, embedding_dim=64, context_size=3):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.word_to_ix = {}
        self.ix_to_word = {}
        self.embeddings = None
        self.context_weights = None
        self.output_weights = None
        
    def build_vocab(self, texts):
        # Count all words
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Keep most common words
        most_common = word_counts.most_common(self.vocab_size - 2)  # -2 for UNK and PAD
        self.word_to_ix = {word: i+2 for i, (word, _) in enumerate(most_common)}
        self.word_to_ix['<PAD>'] = 0
        self.word_to_ix['<UNK>'] = 1
        self.ix_to_word = {i: w for w, i in self.word_to_ix.items()}
        
        # Initialize weights
        self.embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
        self.context_weights = np.random.randn(self.embedding_dim * self.context_size, self.embedding_dim) * 0.1
        self.output_weights = np.random.randn(self.embedding_dim, self.vocab_size) * 0.1
    
    def prepare_sequence(self, text):
        words = text.lower().split()
        return [self.word_to_ix.get(w, self.word_to_ix['<UNK>']) for w in words]
    
    def get_context_vector(self, word_indices):
        vectors = []
        for ix in word_indices:
            vectors.append(self.embeddings[ix])
        return np.concatenate(vectors)
    
    def forward(self, context_indices):
        # Get context embeddings and concatenate
        context_vector = self.get_context_vector(context_indices)
        
        # Apply context weights
        hidden = np.tanh(np.dot(context_vector, self.context_weights))
        
        # Calculate output probabilities
        output = np.dot(hidden, self.output_weights)
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / exp_output.sum()
        
        return probabilities
    
    def train(self, texts, learning_rate=0.01, epochs=5):
        print("Building vocabulary...")
        self.build_vocab(texts)
        
        print("Training model...")
        for epoch in range(epochs):
            total_loss = 0
            num_samples = 0
            
            for text in texts:
                sequence = self.prepare_sequence(text)
                
                for i in range(len(sequence) - self.context_size):
                    context = sequence[i:i + self.context_size]
                    target = sequence[i + self.context_size]
                    
                    # Forward pass
                    probs = self.forward(context)
                    
                    # Calculate cross-entropy loss
                    loss = -np.log(probs[target] + 1e-10)
                    total_loss += loss
                    
                    # Backward pass (simplified gradient descent)
                    d_probs = probs.copy()
                    d_probs[target] -= 1
                    
                    context_vector = self.get_context_vector(context)
                    hidden = np.tanh(np.dot(context_vector, self.context_weights))
                    
                    # Update weights
                    d_output_weights = np.outer(hidden, d_probs)
                    self.output_weights -= learning_rate * d_output_weights
                    
                    d_hidden = np.dot(d_probs, self.output_weights.T)
                    d_context_weights = np.outer(context_vector, d_hidden)
                    self.context_weights -= learning_rate * d_context_weights
                    
                    num_samples += 1
            
            avg_loss = total_loss / num_samples
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    def generate_text(self, seed_text, length=50):
        words = seed_text.lower().split()[-self.context_size:]
        generated = list(words)
        
        for _ in range(length):
            context = [self.word_to_ix.get(w, self.word_to_ix['<UNK>']) for w in words]
            probs = self.forward(context)
            next_word_ix = np.random.choice(len(probs), p=probs)
            next_word = self.ix_to_word[next_word_ix]
            generated.append(next_word)
            words = words[1:] + [next_word]
        
        return ' '.join(generated)
    
    def save(self, filepath):
        model_data = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'context_size': self.context_size,
            'word_to_ix': self.word_to_ix,
            'ix_to_word': self.ix_to_word,
            'embeddings': self.embeddings.tolist(),
            'context_weights': self.context_weights.tolist(),
            'output_weights': self.output_weights.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        model = cls(
            vocab_size=model_data['vocab_size'],
            embedding_dim=model_data['embedding_dim'],
            context_size=model_data['context_size']
        )
        
        model.word_to_ix = model_data['word_to_ix']
        model.ix_to_word = {int(k): v for k, v in model_data['ix_to_word'].items()}
        model.embeddings = np.array(model_data['embeddings'])
        model.context_weights = np.array(model_data['context_weights'])
        model.output_weights = np.array(model_data['output_weights'])
        
        return model
