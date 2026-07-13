from __future__ import annotations

import os

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel


class PlatformWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        platform = os.environ.get("HCP_WHEEL_PLAT_NAME")
        if not platform:
            raise RuntimeError("HCP_WHEEL_PLAT_NAME is required for binary wheels")
        return "py3", "none", platform


setup(cmdclass={"bdist_wheel": PlatformWheel})
