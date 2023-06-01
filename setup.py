import os
import numpy as np
from os.path import join
from setuptools import setup, Extension
from Cython.Build import cythonize


directory_path = os.path.dirname(
    os.path.abspath(__file__)
    )


ext_data = {
    'MixtureOptDesign.coordinate_exchange.coordinate': {
        'sources': [join(directory_path, 'MixtureOptDesign', 'coordinate_exchange', 'coordinate.pyx')],
        'include': [np.get_include()]},
    'MixtureOptDesign.mnl.utils': {
        'sources': [join(directory_path, 'MixtureOptDesign', 'mnl', 'utils.pyx')],
        'include': [np.get_include()]}
    # ,
    # 'MixtureOptDesign.CoordinateExchange.coordinate_cluster': {
    #     'sources': [join(directory_path, 'MixtureOptDesign', 'CoordinateExchange','coordinate_cluster.pyx')],
    #     'include': [np.get_include()]}
    
    ,
    'MixtureOptDesign.vns.vns_cython': {
        'sources': [join(directory_path, 'MixtureOptDesign', 'vns', 'vns_cython.pyx')],
        'include': [np.get_include()]}
    
    }



extensions = []

for name, data in ext_data.items():

    sources = data['sources']
    include = data.get('include', [])

    obj = Extension(
        name,
        sources=sources,
        include_dirs=include
    )
    
    extensions.append(obj)


# Use cythonize on the extension object.
setup(
    name='MixtureOptDesign',
    author='Geoff',
    ext_modules=cythonize(extensions))