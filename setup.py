from setuptools import setup, find_packages

# Define version information
NAME = "gcml"

PACKAGES = [
    "gcml",
]


def search_packages():
    packages = []
    for p in find_packages():
        is_pack = False
        for name in PACKAGES:
            if not p.startswith(name):
                continue
            is_pack = True
            break
        if not is_pack:
            continue
        packages.append(p)
    return packages


setup(
    name=NAME,
    version="0.1",
    description="Stanford CS330 Fall 2021 Goal-Conditioned Meta-Learning",
    packages=search_packages(),
    include_package_data=True,
    zip_safe=False, install_requires=['numpy', 'gym']
)
