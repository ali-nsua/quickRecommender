import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quickrecommender",
    version="0.0.1",
    author="Ali Hassani",
    author_email="alihassanijr1998@gmail.com",
    description="A quick, unsupervised, content-based recommendation system.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ali-nsua/quickRecommender",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)