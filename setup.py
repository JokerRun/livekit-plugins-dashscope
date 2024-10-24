from setuptools import find_namespace_packages, setup

setup(
    name="livekit-plugins-dashscope",
    version="0.1.0", 
    description="DashScope plugin for LiveKit Agents",
    author="Jerry Ruan",
    author_email="565636992@qq.com",
    url="https://github.com/JerryRuan/livekit-plugins-dashscope",
    packages=find_namespace_packages(include=["livekit.*"]),
    install_requires=[
        "livekit-agents>=0.10.0",
        "dashscope>=1.0.0",
    ],
    python_requires=">=3.8",
)
