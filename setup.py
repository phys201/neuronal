from setuptools import setup

setup(name='neuronal',
      version='0.1',
      description='Bayesian inference on neuronal signals',
      url='https://github.com/phys201/neuronal',
      author='Amelia Paine, Han Sae Jung',
      author_email='apaine@g.harvard.edu, hansaejung@g.harvard.edu',
      license='GPL',
      packages=['neuronal'],
      install_requires=[
          'pymc3', 'theano', 'pandas', 'numpy', 'matplotlib' 
      ],
      zip_safe=False)
