# c_backend/setup.py

from setuptools import setup, Extension
import pybind11

# 컴파일러에 전달할 추가 인자들을 정의합니다.
# 원래 명령어에 있던 -O3, -Wall, -std=c++17, -fPIC 플래그를 여기에 포함합니다.
extra_compile_args = ['-std=c++17', '-O3', '-Wall', '-fPIC']

ext_modules = [
    Extension(
        'engine',  # 파이썬에서 import할 모듈 이름
        ['engine.cpp'],  # 컴파일할 소스 파일 목록
        include_dirs=[
            # pybind11 헤더 파일의 경로를 자동으로 찾아줍니다.
            # `python -m pybind11 --includes` 명령어와 동일한 역할을 합니다.
            pybind11.get_include(),
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name='engine',
    version='0.0.1',
    author='Your Name', # 작성자 이름
    author_email='your@email.com', # 작성자 이메일
    description='A pybind11 extension module',
    ext_modules=ext_modules,
)