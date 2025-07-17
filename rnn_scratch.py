import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class CharRNN:
    """
    Character-level Recurrent Neural Network for text generation
    """
    
    def __init__(self, data: str, hidden_size: int = 128, seq_length: int = 30, 
                 learning_rate: float = 0.01, init_scale: float = 0.1):
        """
        Initialize the RNN with given parameters
        
        Args:
            data: Training text data
            hidden_size: Size of hidden layer
            seq_length: Length of input sequences
            learning_rate: Learning rate for parameter updates
            init_scale: Scale for parameter initialization
        """
        self.data = data
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        
        # Character mappings
        self.chars = sorted(list(set(data)))
        self.data_size = len(data)
        self.vocab_size = len(self.chars)
        
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        print(f'Data size: {self.data_size}, Vocab size: {self.vocab_size}')
        print(f'Vocabulary: {self.chars}')
        
        # Initialize parameters with Xavier initialization
        self.Wxh = np.random.randn(hidden_size, self.vocab_size) * np.sqrt(2.0 / self.vocab_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.Why = np.random.randn(self.vocab_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))
        
        # Memory variables for Adagrad
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)
        
        # Training tracking
        self.smooth_loss = -np.log(1.0/self.vocab_size) * self.seq_length
        self.loss_history = []
        
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def forward_backward(self, inputs: List[int], targets: List[int], 
                        hprev: np.ndarray) -> Tuple[float, Dict[str, np.ndarray], np.ndarray]:
        """
        Forward and backward pass through the network
        
        Args:
            inputs: List of input character indices
            targets: List of target character indices
            hprev: Previous hidden state
            
        Returns:
            loss: Cross-entropy loss
            gradients: Dictionary of gradients
            last_hidden: Last hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        
        # Forward pass
        for t in range(len(inputs)):
            # One-hot encode input
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            
            # Hidden state
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + 
                           np.dot(self.Whh, hs[t-1]) + self.bh)
            
            # Output
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = self.softmax(ys[t])
            
            # Cross-entropy loss
            loss += -np.log(ps[t][targets[t], 0])
        
        # Backward pass
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        
        for t in reversed(range(len(inputs))):
            # Output layer gradients
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            
            # Hidden layer gradients
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh  # tanh derivative
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        
        # Gradient clipping
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        gradients = {
            'dWxh': dWxh, 'dWhh': dWhh, 'dWhy': dWhy,
            'dbh': dbh, 'dby': dby
        }
        
        return loss, gradients, hs[len(inputs)-1]
    
    def sample(self, h: np.ndarray, seed_ix: int, n: int, 
               temperature: float = 1.0) -> List[int]:
        """
        Sample a sequence of characters from the model
        
        Args:
            h: Hidden state
            seed_ix: Seed character index
            n: Number of characters to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            List of sampled character indices
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            
            # Apply temperature
            y = y / temperature
            p = self.softmax(y)
            
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
            
        return ixes
    
    def update_parameters(self, gradients: Dict[str, np.ndarray]) -> None:
        """Update parameters using Adagrad optimizer"""
        params = [self.Wxh, self.Whh, self.Why, self.bh, self.by]
        grads = [gradients['dWxh'], gradients['dWhh'], gradients['dWhy'], 
                gradients['dbh'], gradients['dby']]
        mems = [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]
        
        for param, grad, mem in zip(params, grads, mems):
            mem += grad * grad
            param -= self.learning_rate * grad / np.sqrt(mem + 1e-8)
    
    def train(self, num_iterations: int = 50000, print_every: int = 2000, 
              sample_every: int = 5000, sample_length: int = 200) -> None:
        """
        Train the RNN
        
        Args:
            num_iterations: Number of training iterations
            print_every: Print loss every N iterations
            sample_every: Generate samples every N iterations
            sample_length: Length of generated samples
        """
        n = 0
        p = 0
        hprev = np.zeros((self.hidden_size, 1))
        
        print("Starting training...")
        
        for iteration in range(num_iterations):
            # Prepare inputs (sliding window)
            if p + self.seq_length + 1 >= len(self.data) or n == 0:
                hprev = np.zeros((self.hidden_size, 1))
                p = 0
            
            inputs = [self.char_to_ix[ch] for ch in self.data[p:p+self.seq_length]]
            targets = [self.char_to_ix[ch] for ch in self.data[p+1:p+self.seq_length+1]]
            
            # Forward and backward pass
            loss, gradients, hprev = self.forward_backward(inputs, targets, hprev)
            
            # Update parameters
            self.update_parameters(gradients)
            
            # Track smooth loss
            self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
            self.loss_history.append(self.smooth_loss)
            
            # Print progress
            if iteration % print_every == 0:
                print(f'Iteration {iteration}, Loss: {self.smooth_loss:.4f}')
            
            # Generate samples
            if iteration % sample_every == 0:
                print(f'--- Sample at iteration {iteration} ---')
                
                # Try different temperatures
                for temp in [0.5, 0.8, 1.0]:
                    sample_ix = self.sample(hprev, inputs[0], min(100, sample_length), temperature=temp)
                    txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
                    print(f'Temp {temp}: {txt[:100]}...')
                
                print('--- End sample ---\n')
            
            p += self.seq_length
            n += 1
    
    def generate_text(self, seed_text: str, length: int = 1000, 
                     temperature: float = 1.0) -> str:
        """
        Generate text starting with seed text
        
        Args:
            seed_text: Starting text
            length: Length of text to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        h = np.zeros((self.hidden_size, 1))
        
        # Prime the network with seed text
        for ch in seed_text[:-1]:
            x = np.zeros((self.vocab_size, 1))
            x[self.char_to_ix[ch]] = 1
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
        
        # Generate new text
        seed_ix = self.char_to_ix[seed_text[-1]]
        ixes = self.sample(h, seed_ix, length, temperature)
        
        return seed_text + ''.join(self.ix_to_char[ix] for ix in ixes)
    
    def plot_loss(self) -> None:
        """Plot training loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


# Example usage with better training data
def main():
    # Rich, diverse training data with multiple text styles
    data = """The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
    
    Once upon a time, in a kingdom far away, there lived a wise old king who ruled with justice and compassion. His subjects loved him dearly, for he always listened to their concerns and made decisions that benefited everyone.
    
    In the realm of science, we observe that energy cannot be created or destroyed, only transformed from one form to another. This fundamental principle governs everything from the smallest atomic interactions to the largest cosmic phenomena.
    
    "To be or not to be, that is the question," pondered Hamlet as he contemplated the nature of existence. Shakespeare's profound words continue to resonate with audiences centuries after they were written.
    
    The rapid advancement of artificial intelligence has transformed our world in ways we never imagined possible. Machine learning algorithms now power everything from search engines to autonomous vehicles, revolutionizing how we work and live.
    
    Mountains rise majestically against the azure sky, their snow-capped peaks gleaming in the golden sunlight. Rivers carve through valleys below, carrying life-giving water to forests, meadows, and distant seas.
    
    In the bustling city streets, people from all walks of life hurry past each other, each carrying their own stories, dreams, and aspirations. The urban symphony of honking cars, chattering voices, and footsteps creates a unique rhythm of modern life.
    
    Scientists have discovered that the human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. This intricate network enables consciousness, memory, creativity, and all the complex behaviors that make us human.
    
    The ancient philosophers wondered about the nature of reality, truth, and knowledge. Plato's cave allegory suggests that what we perceive as reality might be merely shadows of a deeper truth, challenging us to question our assumptions about the world.
    
    Technology evolves at an exponential rate, with each generation of computers becoming faster, smaller, and more powerful than the last. Moore's Law predicted this trend, though physical limitations may eventually slow this progression.
    
    In the depths of the ocean, mysterious creatures adapted to extreme pressure and darkness have evolved unique characteristics. Bioluminescent organisms create their own light, while others have developed extraordinary hunting strategies.
    
    The art of cooking combines science with creativity, transforming simple ingredients into complex flavors through chemical reactions. Heat denatures proteins, caramelizes sugars, and releases aromatic compounds that delight our senses.
    
    Music has the power to evoke emotions, trigger memories, and bring people together across cultural boundaries. The mathematical relationships between frequencies create harmony and melody that speak to something fundamental in human nature.
    
    Weather patterns across the globe are interconnected in complex ways, with changes in one region affecting conditions thousands of miles away. Climate scientists study these relationships to better understand and predict environmental changes.
    
    Education opens doors to opportunity and understanding, empowering individuals to think critically, solve problems, and contribute meaningfully to society. Knowledge shared across generations builds the foundation for human progress.
    
    The night sky has inspired wonder and curiosity throughout human history. Ancient civilizations mapped constellations, while modern astronomers explore distant galaxies, seeking to understand our place in the vast cosmos.
    
    In gardens and wild spaces, plants and animals form intricate ecosystems where every species plays a vital role. Pollinating insects enable plant reproduction, while decomposers recycle nutrients back into the soil.
    
    Human creativity manifests in countless forms: paintings that capture beauty, stories that transport us to other worlds, inventions that solve problems, and ideas that challenge conventional thinking.
    
    The journey of discovery never ends, as each answer leads to new questions, each solution reveals new problems to solve, and each generation builds upon the knowledge and achievements of those who came before.
    
    Communication bridges the gap between minds, allowing us to share ideas, emotions, and experiences. Language evolves continuously, adapting to new technologies and cultural changes while preserving the wisdom of the past.
    
    In laboratories around the world, researchers push the boundaries of human knowledge, conducting experiments that may lead to breakthrough discoveries in medicine, physics, chemistry, and countless other fields.
    
    The balance between tradition and innovation shapes every culture, as societies honor their heritage while adapting to changing circumstances and embracing new possibilities for the future.
    
    Dreams and aspirations drive human achievement, motivating individuals to overcome obstacles, pursue excellence, and create something meaningful that will outlast their own brief existence on this planet.
    
    The interconnectedness of all things becomes apparent when we consider how small actions can have far-reaching consequences, how local events affect global systems, and how the past shapes the present and future.
    
    In moments of quiet reflection, we might glimpse the profound mystery of consciousness itself: the fact that matter organized in a particular way can experience, feel, think, and contemplate its own existence.
    
    Time flows like a river, carrying us forward through experiences that shape who we become. Each moment offers choices that ripple outward, creating the complex tapestry of human history and individual destiny.
    
    The pursuit of happiness, truth, and meaning defines the human experience, driving us to explore, create, love, and strive for something greater than ourselves in this remarkable journey called life."""
    
    # Create and train the model with better parameters
    rnn = CharRNN(data, hidden_size=128, seq_length=30, learning_rate=0.01)
    rnn.train(num_iterations=50000, print_every=2000, sample_every=5000)
    
    # Generate text with different approaches
    print("\n" + "="*50)
    print("FINAL GENERATED TEXT:")
    print("="*50)
    
    # Try different starting seeds and temperatures
    seeds = ["The", "In", "Science", "Once upon"]
    temperatures = [0.6, 0.8, 1.0]
    
    for seed in seeds:
        print(f"\n--- Starting with: '{seed}' ---")
        for temp in temperatures:
            try:
                generated = rnn.generate_text(seed, length=150, temperature=temp)
                print(f"Temp {temp}: {generated}")
            except KeyError:
                print(f"Temp {temp}: Seed '{seed}' not in vocabulary, using 'T' instead")
                generated = rnn.generate_text("T", length=150, temperature=temp)
                print(f"Temp {temp}: {generated}")
        print()
    
    # Plot training loss
    rnn.plot_loss()


if __name__ == "__main__":
    main()