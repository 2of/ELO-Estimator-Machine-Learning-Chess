import tensorflow as tf


#From copilot... just checking that the tf install is happy chappy (Spoilers, linux hates me on this one)

def check_tensorflow():
    """
    Checks if TensorFlow is installed and working by performing a simple operation.
    """
    print("Checking TensorFlow installation...")
    try:
        # Print TensorFlow version
        print(f"TensorFlow version: {tf.__version__}")

        # Perform a simple operation to test if it's functional
        hello_tensor = tf.constant("Hello, TensorFlow!")
        tf.print(hello_tensor)

        # Basic tensor operation
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("\nPerforming a matrix multiplication (a * b):")
        print("Tensor 'a':\n", a.numpy())
        print("Tensor 'b':\n", b.numpy())
        print("Result 'c' (a * b):\n", c.numpy())

        print("\nTensorFlow is working correctly! 🎉")

    except ImportError:
        print("Error: TensorFlow is not installed. Please install it using 'pip install tensorflow'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("TensorFlow might be installed but encountering issues. Check your environment setup.")

if __name__ == "__main__":
    check_tensorflow()