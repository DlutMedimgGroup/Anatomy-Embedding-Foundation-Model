from setuptools import find_packages, setup

setup(
    name="medsam",
    version="0.0.1",
    author="Jun Ma",
    python_requires=">=3.9",
    install_requires=[
        "monai", 
        "SimpleITK>=2.2.1", 
        "tqdm", 
        "toml"
        ],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["pycocotools", "opencv-python", "onnx", "onnxruntime"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)