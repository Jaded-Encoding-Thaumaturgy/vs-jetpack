#!/usr/bin/env python3

import setuptools
from pathlib import Path

package_name = 'vsjetpack'

exec(Path('_metadata.py').read_text(), meta := dict[str, str]())

readme = Path('README.md').read_text()
requirements = Path('requirements.txt').read_text()


setuptools.setup(
    name=package_name,
    version=meta['__version__'],
    author=meta['__author_name__'],
    author_email=meta['__author_email__'],
    maintainer=meta['__maintainer_name__'],
    maintainer_email=meta['__maintainer_email__'],
    description=meta['__doc__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    project_urls={
        'Source Code': 'https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack',
        'Contact': 'https://discord.gg/XTpc6Fa9eB',
    },
    install_requires=requirements,
    python_requires='>=3.10',
    packages=[
        'vstools',
        'vstools.enums',
        'vstools.exceptions',
        'vstools.functions',
        'vstools.types',
        'vstools.utils',

        'vskernels',
        'vskernels.abstract',
        'vskernels.kernels',
        'vskernels.kernels.custom',
        'vskernels.kernels.zimg',

        'vsexprtools',

        'vsrgtools',
        'vsrgtools.aka_expr',

        'vsmasktools',
        'vsmasktools.edge',

        'vsaa',

        'vsscale',

        'vsdenoise',
        'vsdenoise.mvtools',

        'vsdehalo',
        'vsdeband',
        'vsdeinterlace',

        'vssource',
        'vssource.indexers',
        'vssource.formats',
        'vssource.formats.bd',
        'vssource.formats.dvd',
        'vssource.formats.dvd.parsedvd',
    ],
    package_data={
        'vstools': ['utils/*.json']
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
