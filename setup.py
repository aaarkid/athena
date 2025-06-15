from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    name="athena-rl",
    version="0.1.0",
    description="High-performance reinforcement learning library written in Rust with Python bindings",
    author="Athena Contributors",
    rust_extensions=[
        RustExtension(
            "athena_py",
            binding=Binding.PyO3,
            features=["python"],
            debug=False,
        )
    ],
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "dev": ["pytest", "pytest-benchmark"],
    },
    zip_safe=False,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
    ],
)