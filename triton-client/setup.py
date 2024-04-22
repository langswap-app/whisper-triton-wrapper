from setuptools import find_packages, setup

setup(
    name="tts_client",
    version="0.0.1",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=[
        "numpy",
        "tritonclient[all]",
    ],
)
