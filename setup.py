from setuptools import setup, find_packages

setup(
    name="MaskAD",
    version="0.1.0",
    author="Tianle Liu",             # 比如 "Tianle Liu"
    author_email="liutianle@zju.edu.cn",  # 可选，可以先不写或者写个占位
    description="Mask-based autonomous driving planner for nuPlan.",
    long_description="MaskAD: A mask-guided diffusion planner for autonomous driving.",
    long_description_content_type="text/plain",  # 如果用 README.md，可以用 "text/markdown"
    url="",
    license="MIT",                           # 或者 "Apache-2.0" / "BSD-3-Clause" / "Proprietary" 等
    packages=find_packages(),                # 自动发现 MaskAD 下所有包
    include_package_data=True,               # 允许包含 yaml/json 等非 py 文件
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",  # 如果不用 MIT，就换掉这一行
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
