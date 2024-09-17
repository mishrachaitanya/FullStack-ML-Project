from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path:str)->List[str]:

    #    Returns the list of requirements
    requirements=[]
    with open(file_path ) as f:
        requirements=f.readlines()
        requirements = [requirement.replace("/n","") for requirement in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    description='Machine Learning Project-Full Stack',
    author='ChaitanyaMishra',
    author_email='22f1000748@ds.study.iitm.ac.in',
    packages=find_packages(),
    install_requires=get_requirements('./requirements.txt')
)
