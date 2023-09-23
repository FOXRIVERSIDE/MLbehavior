import math
import numpy as np

class Complex:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        return Complex(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        return Complex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )

    def magnitude(self):
        return math.sqrt(self.real * self.real + self.imag * self.imag)
def perform_radix2_fft(input_data):
    # Find the smallest power of 2 that is greater than or equal to the input size
    n = len(input_data)
    fft_size = 2 ** math.ceil(math.log2(n))

    # Zero-pad the input to the next power of 2
    padded_input = input_data.tolist() + [0.0] * (fft_size - n)

    # Recursive radix-2 FFT implementation
    def radix2_fft(input_data, inverse=False):
        n = len(input_data)
        fft_size = 2 ** math.ceil(math.log2(n))

        # Zero-pad the input to the next power of 2
        padded_input = input_data + [0.0] * (fft_size - n)

        # Base case: if the input size is 1, return the input as is
        if n == 1:
            return input_data

        # Split the input into even and odd parts
        even = [input_data[i] for i in range(n) if i % 2 == 0]
        odd = [input_data[i] for i in range(n) if i % 2 != 0]

        # Compute FFT of even and odd parts recursively
        even_fft = radix2_fft(even, inverse)
        odd_fft = radix2_fft(odd, inverse)

        # Combine even and odd parts with twiddle factors
        twiddle_sign = 1 if inverse else -1
        output = [Complex(0.0, 0.0)] * n
        for k in range(n // 2):
            twiddle = Complex(math.cos(twiddle_sign * 2.0 * math.pi * k / n),
                              math.sin(twiddle_sign * 2.0 * math.pi * k / n))
            odd_term = twiddle * odd_fft[k]
            output[k] = even_fft[k] + odd_term
            output[k + n // 2] = even_fft[k] - odd_term

        return output

    # Convert the input to complex format (real with imaginary parts set to 0)
    complex_input = [Complex(data, 0.0) for data in padded_input]

    # Perform the FFT
    fft_result = radix2_fft(complex_input, inverse=False)

    # Calculate magnitudes from the complex FFT result
    magnitudes = [complex_data.magnitude() for complex_data in fft_result]

    return magnitudes



