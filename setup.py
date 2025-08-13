from setuptools import setup, find_packages
import os


def read_requirements():
    """Read dependencies from the root requirements.txt"""
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.exists(req_file):
        # Fallback: use live_ai_interpreter/backend/requirements.txt if present
        alt = os.path.join(os.path.dirname(__file__), "live_ai_interpreter", "backend", "requirements.txt")
        req_file = alt if os.path.exists(alt) else None
    if not req_file:
        return []
    with open(req_file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="meeting-live-language",
    version="0.1.0",
    author="Siddhanath",
    author_email="siddhanath.hanna@gmail.com",
    description="A real-time live meeting language interpreter system with speech-to-text, translation, and speech synthesis.",
    long_description=(
        "Meeting Live Language Project\n"
        "Author: Siddhanath\n"
        "Email: siddhanath.hanna@gmail.com\n"
        "Phone: 7705923690\n\n"
        "This package provides a live AI interpreter for multilingual meetings."
    ),
    long_description_content_type="text/plain",
    url="",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.12",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)



