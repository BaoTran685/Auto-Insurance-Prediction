import sys
import os

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout  # Save the original stdout
        sys.stdout = open(os.devnull, 'w')  # Redirect stdout to null

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()  # Close the null file
        sys.stdout = self._original_stdout  # Restore original stdout
        