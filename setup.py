# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
    ['imdb_sentiment', 'imdb_sentiment.models', 'imdb_sentiment.shared.utils']

package_data = \
    {'': ['*']}

install_requires = \
    ['pydantic>=1.10.7,<2.0.0']

entry_points = \
    {'console_scripts': ['imdb_sentiment = imdb_sentiment:main',
                         'test = pytest:main']}

setup_kwargs = {
    'name': 'imdb-sentiment',
    'version': '0.1.0',
    'description': '',
    'long_description': '',
    'author': 'ZappaBoy',
    'author_email': 'federico.zappone@justanother.cloud',
    'maintainer': 'ZappaBoy',
    'maintainer_email': 'federico.zappone@justanother.cloud',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.11,<3.12',
}

setup(**setup_kwargs)
