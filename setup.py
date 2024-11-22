from setuptools import setup, find_packages

setup(
    name="causal_trading",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'pandas',
        'numpy',
        'dowhy',
        'econml',
        'networkx',
        'scikit-learn',
        'pytest'
    ],
)
