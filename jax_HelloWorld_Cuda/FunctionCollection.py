import jax.numpy as jnp
import jax
@jax.jit
def runtime_example(x):
    """Computes the dot product of a matrix with its transpose.
    
    Args:
        x (jax.Array): Input matrix
        
    Returns:
        jax.Array: Result of exp(x @ x.T)
        
    Example:
        >>> x = jnp.array([[1., 2.], [3., 4.]])
        >>> runtime_example(x)
    """
    return jnp.dot(x, x.T)

@jax.jit
def chained_runtime_examples(x):
    """Applies runtime_example thrice.
    
    Args:
        x (jax.Array): Input matrix
        
    Returns:
        jax.Array: Result of runtime_example(runtime_example(x))
        
    Example:
        >>> x = jnp.array([[1., 2.], [3., 4.]])
        >>> chained_runtime_examples(x)
    """
    return runtime_example(runtime_example(runtime_example(x)))



def glorot_uniform(key, shape, dtype=jnp.float32):
    """
    Glorot/Xavier uniform initialization.
    """
    limit = jnp.sqrt(6 / sum(shape))
    return jax.random.uniform(key, shape, dtype, -limit, limit)

def glorot_normal(key, shape, dtype=jnp.float32):
    """
    Glorot/Xavier normal initialization.
    """
    std = jnp.sqrt(2 / sum(shape))
    return jax.random.normal(key, shape, dtype) * std

import time
from IPython.display import HTML
class PlotDisplay:
    def __init__(self):
        self.timestamp = 0
        
    def update(self):
        self.timestamp = int(time.time() * 1000)
        
    def show(self, filename, width=600, height=None):
        if height is None:
            #16-9 aspect ratio by default
            height = width/16*9
        return HTML(f'''
        <div style="text-align: center">
            <img src="{filename}?v={self.timestamp}" width="{width}"/>
        </div>
        ''')
