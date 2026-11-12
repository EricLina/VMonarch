from setuptools import setup, find_packages

setup(
    name="video_monarch_attention",
    version="0.1.0",
    description="VMonarch: Efficient Video Diffusion Transformers with Structured Attention",
    author="Cheng Liang",
    author_email="Lc1190301804@outlook.com",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.7.0",
        "triton>=3.5.1",
    ],
)
